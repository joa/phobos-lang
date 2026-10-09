// Times a `<fmt>_qgemm` kernel from a PTX or cubin file in isolation:
// device-resident operands, the push_descriptor launch ABI, a 512-row prompt
// at a model's projection shapes. Reports TOPS and the per-launch time.
//
//     gcc -O2 -o qg qg.c -I. -lcuda && ./qg KERNEL.ptx NAME BLOCK_BYTES [ROWS]
//
// BLOCK_BYTES is the format's bytes per 256-element block on the device,
// 144 for Q4_K, 176 for Q5_K, 208 for Q6_K.
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

static char *read_file(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); exit(1); }
    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *buf = calloc(1, len + 1);
    if (fread(buf, 1, len, f) != (size_t)len) { perror("read"); exit(1); }
    fclose(f);
    if (len > 4 && buf[0] == 0x7f && buf[1] == 'E') return buf;
    for (long i = 0; i + 1 < len; i++)
        if (buf[i] == '/' && buf[i + 1] == '/') return buf + i;
    return buf;
}

static CUdeviceptr filled(size_t bytes, unsigned char byte) {
    CUdeviceptr d;
    CK(cuMemAlloc(&d, bytes));
    CK(cuMemsetD8(d, byte, bytes));
    return d;
}

static void push(unsigned long long *slots, int *at, CUdeviceptr ptr, long d0, long d1) {
    unsigned long long v[7] = {ptr, ptr, 0, (unsigned long long)d0, (unsigned long long)d1, (unsigned long long)d1, 1};
    for (int i = 0; i < 7; i++) slots[(*at)++] = v[i];
}

int main(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "usage: qg KERNEL NAME BLOCK_BYTES [ROWS]\n"); return 2; }
    long block_bytes = atol(argv[3]), m = argc > 4 ? atol(argv[4]) : 512;
    CK(cuInit(0));
    CUdevice dev;
    CK(cuDeviceGet(&dev, 0));
    CUcontext ctx;
    CK(cuDevicePrimaryCtxRetain(&ctx, dev));
    CK(cuCtxSetCurrent(ctx));
    CUmodule mod;
    CK(cuModuleLoadData(&mod, read_file(argv[1])));
    CUfunction fn;
    CK(cuModuleGetFunction(&fn, mod, argv[2]));
    int regs, local, shared;
    CK(cuFuncGetAttribute(&regs, CU_FUNC_ATTRIBUTE_NUM_REGS, fn));
    CK(cuFuncGetAttribute(&local, CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, fn));
    CK(cuFuncGetAttribute(&shared, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, fn));
    int per_sm = 0;
    CK(cuOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, fn, 256, 0));
    printf("%s: %d regs, %d B local, %d B static shared, %d CTAs/SM\n", argv[2], regs, local, shared, per_sm);
    CUevent e0, e1;
    CK(cuEventCreate(&e0, 0));
    CK(cuEventCreate(&e1, 0));
    // (n, k) of the projections: hidden 2560 and ffn 9216 for the 4B.
    long shapes[][2] = {{2560, 2560}, {9216, 2560}, {2560, 9216}};
    double total_ops = 0, total_us = 0;
    for (unsigned s = 0; s < sizeof shapes / sizeof *shapes; s++) {
        long n = shapes[s][0], k = shapes[s][1];
        long nb = k / 256, kb = k / 32, rb = nb * block_bytes;
        // Bytes 0x11 keep every f16 and f32 scale a small finite number.
        CUdeviceptr a = filled((size_t)m * k, 0x11), as = filled((size_t)m * kb * 4, 0x11),
                    qb = filled((size_t)n * rb, 0x11), d = filled((size_t)n * nb * 2, 0x11), c;
        CK(cuMemAlloc(&c, (size_t)m * n * 4));
        unsigned long long slots[35];
        int at = 0;
        push(slots, &at, a, m, k);
        push(slots, &at, as, m, kb);
        push(slots, &at, qb, n, rb);
        push(slots, &at, d, n, nb);
        push(slots, &at, c, m, n);
        void *args[35];
        for (int i = 0; i < 35; i++) args[i] = &slots[i];
        unsigned gx = (unsigned)(m / 128), gy = (unsigned)(n / 64);
        CK(cuLaunchKernel(fn, gx, gy, 1, 256, 1, 1, 0, 0, args, NULL));
        CK(cuCtxSynchronize());
        const int reps = 20;
        CK(cuEventRecord(e0, 0));
        for (int i = 0; i < reps; i++) CK(cuLaunchKernel(fn, gx, gy, 1, 256, 1, 1, 0, 0, args, NULL));
        CK(cuEventRecord(e1, 0));
        CK(cuEventSynchronize(e1));
        float ms;
        CK(cuEventElapsedTime(&ms, e0, e1));
        double us = 1000.0 * ms / reps, ops = 2.0 * m * n * k;
        total_ops += ops, total_us += us;
        printf("  m %ld n %5ld k %5ld: %4u CTAs, %8.1f us, %6.1f TOPS\n", m, n, k, gx * gy, us, ops / (us * 1e-6) / 1e12);
        cuMemFree(a), cuMemFree(as), cuMemFree(qb), cuMemFree(d), cuMemFree(c);
    }
    printf("  all three: %.1f TOPS\n", total_ops / (total_us * 1e-6) / 1e12);
    return 0;
}
