// Times phobos's real q4k_qdot_i8_matvec, loaded from a kernel-cache file, in
// isolation: device-resident weights and activations, the launch ABI of
// phobos-kernels' push_descriptor, one launch per projection shape.
//
//     gcc -O2 -o kq kq.c -I. -lcuda && ./kq CACHE_FILE TN [TN ...]
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CK(x)                                                                                                         \
    do {                                                                                                              \
        CUresult e_ = (x);                                                                                            \
        if (e_ != CUDA_SUCCESS) {                                                                                     \
            const char *s_ = "?";                                                                                     \
            cuGetErrorName(e_, &s_);                                                                                  \
            fprintf(stderr, "%s:%d %s -> %s\n", __FILE__, __LINE__, #x, s_);                                          \
            exit(1);                                                                                                  \
        }                                                                                                             \
    } while (0)

static char *read_ptx(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); exit(1); }
    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *buf = calloc(1, len + 1);
    if (fread(buf, 1, len, f) != (size_t)len) { perror("read"); exit(1); }
    fclose(f);
    // A cubin loads as is.
    if (len > 4 && buf[0] == 0x7f && buf[1] == 'E') return buf;
    // The cache file carries a short binary header before the PTX text.
    for (long i = 0; i + 1 < len; i++)
        if (buf[i] == '/' && buf[i + 1] == '/') return buf + i;
    fprintf(stderr, "no PTX in %s\n", path);
    exit(1);
}

static CUdeviceptr upload(const void *host, size_t bytes) {
    CUdeviceptr d;
    CK(cuMemAlloc(&d, bytes));
    CK(cuMemcpyHtoD(d, host, bytes));
    return d;
}

// One operand as push_descriptor lays it out.
static void push(unsigned long long *slots, int *at, CUdeviceptr ptr, long d0, long d1) {
    unsigned long long w0 = (unsigned)(int)d0, w1 = (unsigned)(int)d1;
    unsigned long long v[7] = {ptr, ptr, 0, w0, w1, w1, 1};
    for (int i = 0; i < 7; i++) slots[(*at)++] = v[i];
}

static unsigned short f16(float x) {
    // Enough for small positive normals.
    unsigned u;
    memcpy(&u, &x, 4);
    unsigned exp = ((u >> 23) & 0xff) - 127 + 15, man = (u >> 13) & 0x3ff;
    return (unsigned short)((exp << 10) | man);
}

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: kq CACHE_FILE TN\n"); return 2; }
    const char *path = argv[1];
    int tn = atoi(argv[2]);
    CK(cuInit(0));
    CUdevice dev;
    CK(cuDeviceGet(&dev, 0));
    CUcontext ctx;
    CK(cuDevicePrimaryCtxRetain(&ctx, dev));
    CK(cuCtxSetCurrent(ctx));
    CUmodule mod;
    CK(cuModuleLoadData(&mod, read_ptx(path)));
    CUfunction fn;
    CK(cuModuleGetFunction(&fn, mod, "q4k_qdot_i8_matvec"));
    int regs, local;
    CK(cuFuncGetAttribute(&regs, CU_FUNC_ATTRIBUTE_NUM_REGS, fn));
    CK(cuFuncGetAttribute(&local, CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, fn));
    printf("%s TN=%d: regs %d, local %d B\n", strrchr(path, '/') ? strrchr(path, '/') + 1 : path, tn, regs, local);
    CUevent e0, e1;
    CK(cuEventCreate(&e0, 0));
    CK(cuEventCreate(&e1, 0));

    // (n, k) of the 4B's Q4_K projections: hidden 2560, ffn 9216.
    long shapes[][2] = {{2560, 2560}, {9216, 2560}, {2560, 9216}, {12288, 2560}};
    srand(1);
    for (unsigned s = 0; s < sizeof shapes / sizeof *shapes; s++) {
        long n = shapes[s][0], k = shapes[s][1];
        long nb = k / 256, kb = k / 32, rb = nb * 144;
        // RB pads the row stride the kernel is told, with the same blocks per row.
        if (getenv("RB") && atol(getenv("RB")) >= rb) rb = atol(getenv("RB"));
        size_t qb_bytes = (size_t)n * rb;
        signed char *aq = malloc(k);
        float *as = malloc(kb * sizeof(float));
        unsigned char *qb = malloc(qb_bytes);
        unsigned short *d = malloc((size_t)n * nb * 2);
        for (long i = 0; i < k; i++) aq[i] = (signed char)(rand() % 255 - 127);
        for (long i = 0; i < kb; i++) as[i] = 0.01f;
        for (size_t i = 0; i < qb_bytes; i++) qb[i] = (unsigned char)rand();
        // Each 144-byte block starts with f16 d and dmin; keep them finite.
        for (size_t b = 0; b < qb_bytes / 144; b++) {
            unsigned short h = f16(0.001f);
            memcpy(qb + b * 144, &h, 2);
            memcpy(qb + b * 144 + 2, &h, 2);
        }
        for (long i = 0; i < n * nb; i++) d[i] = f16(1.0f);
        // Room for one activation copy per block (16 KiB and 2 KiB apart), for
        // the private-activation variant; the others read the first copy.
        signed char *aq_all = calloc(256, 16384);
        float *as_all = calloc(256, 2048);
        for (int c = 0; c < 256; c++) {
            memcpy(aq_all + (size_t)c * 16384, aq, k);
            memcpy((char *)as_all + (size_t)c * 2048, as, kb * 4);
        }
        CUdeviceptr daq = upload(aq_all, 256 * 16384), das = upload(as_all, 256 * 2048), dqb = upload(qb, qb_bytes),
                    dd = upload(d, (size_t)n * nb * 2), dc;
        CK(cuMemAlloc(&dc, n * 4));

        unsigned long long slots[35];
        int at = 0;
        push(slots, &at, daq, 1, k);
        push(slots, &at, das, 1, kb);
        push(slots, &at, dqb, n, rb);
        push(slots, &at, dd, n, nb);
        push(slots, &at, dc, 1, n);
        void *args[35];
        for (int i = 0; i < 35; i++) args[i] = &slots[i];

        unsigned full = (unsigned)(n / tn);
        unsigned grids[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 16, 24, 32, full};
        for (unsigned gi = 0; gi < sizeof grids / sizeof *grids; gi++) {
        unsigned grid = grids[gi];
        // The sweep only on the widest shape; the others at their full grid.
        if (grid > full || (grid != full && s + 1 != sizeof shapes / sizeof *shapes)) continue;
        CK(cuLaunchKernel(fn, grid, 1, 1, 256, 1, 1, 0, 0, args, NULL));
        CK(cuCtxSynchronize());
        const int reps = 50;
        CK(cuEventRecord(e0, 0));
        for (int i = 0; i < reps; i++) CK(cuLaunchKernel(fn, grid, 1, 1, 256, 1, 1, 0, 0, args, NULL));
        CK(cuEventRecord(e1, 0));
        CK(cuEventSynchronize(e1));
        float ms;
        CK(cuEventElapsedTime(&ms, e0, e1));
        double us = 1000.0 * ms / reps;
        printf("  n %5ld k %5ld: %4u blocks, %8.1f us/launch, %6.2f us/block, %7.1f GB/s of weight read\n", n, k,
               grid, us, us / grid, (double)qb_bytes * grid / full / (us * 1e-6) / 1e9);
        }
        float c0;
        CK(cuMemcpyDtoH(&c0, dc, 4));
        cuMemFree(daq), cuMemFree(das), cuMemFree(dqb), cuMemFree(dd), cuMemFree(dc);
        free(aq), free(as), free(qb), free(d), free(aq_all), free(as_all);
    }
    return 0;
}
