// Times a `<fmt>_qgemm` kernel from a PTX or cubin file in isolation:
// device-resident operands, the push_descriptor launch ABI, a 512-row prompt
// at a model's projection shapes. Reports TOPS and the per-launch time.
//
//     gcc -O2 -o qg qg.c -I. -lcuda && ./qg KERNEL.ptx NAME BLOCK_BYTES [ROWS [REF.ptx]]
//
// BLOCK_BYTES is the format's bytes per 256-element block on the device,
// 144 for Q4_K, 176 for Q5_K, 208 for Q6_K. With REF.ptx both kernels run on
// the same random operands and the worst difference is printed: a numeric
// check of a new code path against the one it replaces.
//
// QG_SPLIT_DIR=DIR with QG_SPLITS="2 3 4" also times the split-K form of each
// shape: DIR/<fmt>_split_n<N>_k<K>_s<S>.ptx (kernel NAME_split, partials as
// one [M, S * N] output, the split in grid y) followed by
// DIR/reduce_n<N>_s<S>.ptx (q8_qmma_reduce), and checks their sum against
// KERNEL. A split count with no kernel for a shape is skipped.
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

// Random bytes, with the f16 pair every `stride` bytes kept a small finite
// number (0x2000 to 0x23ff), for the scales a block leads with.
static CUdeviceptr random_bytes(size_t bytes, long stride) {
    unsigned char *h = malloc(bytes);
    for (size_t i = 0; i < bytes; i++) h[i] = (unsigned char)(rand() >> 7);
    if (stride)
        for (size_t at = 0; at + 4 <= bytes; at += stride)
            for (int b = 0; b < 4; b += 2) h[at + b] = (unsigned char)rand(), h[at + b + 1] = 0x20 | (rand() & 3);
    CUdeviceptr d;
    CK(cuMemAlloc(&d, bytes));
    CK(cuMemcpyHtoD(d, h, bytes));
    free(h);
    return d;
}

static CUdeviceptr scales_f32(size_t count) {
    float *h = malloc(count * 4);
    for (size_t i = 0; i < count; i++) h[i] = 0.005f + 0.01f * (float)(rand() % 100) / 100.0f;
    CUdeviceptr d;
    CK(cuMemAlloc(&d, count * 4));
    CK(cuMemcpyHtoD(d, h, count * 4));
    free(h);
    return d;
}

static void push(unsigned long long *slots, int *at, CUdeviceptr ptr, long d0, long d1) {
    unsigned long long v[7] = {ptr, ptr, 0, (unsigned long long)d0, (unsigned long long)d1, (unsigned long long)d1, 1};
    for (int i = 0; i < 7; i++) slots[(*at)++] = v[i];
}

static CUfunction load_fn(const char *path, const char *name) {
    CUmodule mod;
    CK(cuModuleLoadData(&mod, read_file(path)));
    CUfunction fn;
    CK(cuModuleGetFunction(&fn, mod, name));
    return fn;
}

static float time_us(CUfunction fn, unsigned gx, unsigned gy, unsigned gz, unsigned threads, void **args, int reps) {
    CUevent e0, e1;
    CK(cuEventCreate(&e0, 0));
    CK(cuEventCreate(&e1, 0));
    CK(cuEventRecord(e0, 0));
    for (int i = 0; i < reps; i++) CK(cuLaunchKernel(fn, gx, gy, gz, threads, 1, 1, 0, 0, args, NULL));
    CK(cuEventRecord(e1, 0));
    CK(cuEventSynchronize(e1));
    float ms;
    CK(cuEventElapsedTime(&ms, e0, e1));
    cuEventDestroy(e0), cuEventDestroy(e1);
    return 1000.0f * ms / reps;
}

static double worst_rel(const float *got, const float *want, size_t count) {
    double worst = 0, top = 0;
    for (size_t i = 0; i < count; i++) {
        double w = want[i] < 0 ? -want[i] : want[i], e = got[i] - want[i];
        e = e < 0 ? -e : e;
        if (e != e) e = 1e30;
        top = w > top ? w : top;
        worst = e > worst ? e : worst;
    }
    return top > 0 ? worst / top : worst;
}

int main(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "usage: qg KERNEL NAME BLOCK_BYTES [ROWS]\n"); return 2; }
    long block_bytes = atol(argv[3]), m = argc > 4 ? atol(argv[4]) : 512;
    const char *ref_path = argc > 5 ? argv[5] : NULL;
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
    CUfunction ref = NULL;
    if (ref_path) {
        CUmodule ref_mod;
        CK(cuModuleLoadData(&ref_mod, read_file(ref_path)));
        CK(cuModuleGetFunction(&ref, ref_mod, argv[2]));
    }
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
    // QG_SHAPES="N,K N,K ..." replaces them, for another model's or
    // test-backend-ops' shapes.
    long shapes[16][2] = {{2560, 2560}, {9216, 2560}, {2560, 9216}};
    unsigned count = 3;
    const char *custom = getenv("QG_SHAPES");
    if (custom) {
        int used;
        for (count = 0; count < 16 && sscanf(custom, "%ld,%ld%n", &shapes[count][0], &shapes[count][1], &used) == 2;
             count++)
            custom += used;
    }
    double total_ops = 0, total_us = 0;
    for (unsigned s = 0; s < count; s++) {
        long n = shapes[s][0], k = shapes[s][1];
        long nb = k / 256, kb = k / 32, rb = nb * block_bytes;
        CUdeviceptr a = random_bytes((size_t)m * k, 0), as = scales_f32((size_t)m * kb),
                    qb = random_bytes((size_t)n * rb, block_bytes), d = random_bytes((size_t)n * nb * 2, 2), c;
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
        printf("  m %ld n %5ld k %5ld: %4u CTAs, %8.1f us, %6.1f TOPS", m, n, k, gx * gy, us, ops / (us * 1e-6) / 1e12);
        if (ref) {
            size_t count = (size_t)m * n;
            float *got = malloc(count * 4), *want = malloc(count * 4);
            CK(cuLaunchKernel(fn, gx, gy, 1, 256, 1, 1, 0, 0, args, NULL));
            CK(cuMemcpyDtoH(got, c, count * 4));
            CK(cuLaunchKernel(ref, gx, gy, 1, 256, 1, 1, 0, 0, args, NULL));
            CK(cuMemcpyDtoH(want, c, count * 4));
            double worst = 0, top = 0;
            for (size_t i = 0; i < count; i++) {
                double w = want[i] < 0 ? -want[i] : want[i], e = got[i] - want[i];
                e = e < 0 ? -e : e;
                if (e != e) e = 1e30;
                top = w > top ? w : top;
                worst = e > worst ? e : worst;
            }
            printf(", worst diff %.3g of max |ref| %.3g (%.2e)", worst, top, top > 0 ? worst / top : worst);
            free(got), free(want);
        }
        printf("\n");
        const char *split_dir = getenv("QG_SPLIT_DIR"), *splits = getenv("QG_SPLITS");
        int used;
        for (int sp; split_dir && splits && sscanf(splits, "%d%n", &sp, &used) == 1; splits += used) {
            char path[512], name[128];
            snprintf(path, sizeof path, "%s/%.*s_split_n%ld_k%ld_s%d.ptx", split_dir,
                     (int)(strchr(argv[2], '_') - argv[2]), argv[2], n, k, sp);
            FILE *probe = fopen(path, "rb");
            if (!probe) continue;
            fclose(probe);
            snprintf(name, sizeof name, "%s_split", argv[2]);
            CUfunction split = load_fn(path, name);
            snprintf(path, sizeof path, "%s/reduce_n%ld_s%d.ptx", split_dir, n, sp);
            CUfunction reduce = load_fn(path, "q8_qmma_reduce");
            CUdeviceptr part;
            unsigned long long ss[128], rs[128];
            int sat = 0, rat = 0;
            push(ss, &sat, a, m, k);
            push(ss, &sat, as, m, kb);
            push(ss, &sat, qb, n, rb);
            push(ss, &sat, d, n, nb);
            CK(cuMemAlloc(&part, (size_t)m * n * sp * 4));
            push(ss, &sat, part, m, n * sp);
            push(rs, &rat, part, m, n * sp);
            push(rs, &rat, c, m, n);
            void *sargs[128], *rargs[128];
            for (int i = 0; i < 128; i++) sargs[i] = &ss[i], rargs[i] = &rs[i];
            unsigned rx = (unsigned)m, ry = (unsigned)(n / 64);
            CK(cuLaunchKernel(split, gx, gy * sp, 1, 256, 1, 1, 0, 0, sargs, NULL));
            CK(cuLaunchKernel(reduce, rx, ry, 1, 128, 1, 1, 0, 0, rargs, NULL));
            CK(cuCtxSynchronize());
            float split_us = time_us(split, gx, gy * sp, 1, 256, sargs, reps);
            float reduce_us = time_us(reduce, rx, ry, 1, 128, rargs, reps);
            CUevent e2, e3;
            CK(cuEventCreate(&e2, 0));
            CK(cuEventCreate(&e3, 0));
            CK(cuEventRecord(e2, 0));
            for (int i = 0; i < reps; i++) {
                CK(cuLaunchKernel(split, gx, gy * sp, 1, 256, 1, 1, 0, 0, sargs, NULL));
                CK(cuLaunchKernel(reduce, rx, ry, 1, 128, 1, 1, 0, 0, rargs, NULL));
            }
            CK(cuEventRecord(e3, 0));
            CK(cuEventSynchronize(e3));
            float both_ms;
            CK(cuEventElapsedTime(&both_ms, e2, e3));
            double both_us = 1000.0 * both_ms / reps;
            size_t count = (size_t)m * n;
            float *got = malloc(count * 4), *want = malloc(count * 4);
            CK(cuMemcpyDtoH(got, c, count * 4));
            CK(cuLaunchKernel(fn, gx, gy, 1, 256, 1, 1, 0, 0, args, NULL));
            CK(cuMemcpyDtoH(want, c, count * 4));
            printf("    split %d: %5u CTAs, split %7.1f us + reduce %6.1f us = %7.1f us, %6.1f TOPS (%+.0f%%), diff %.2e\n",
                   sp, gx * gy * sp, split_us, reduce_us, both_us, ops / (both_us * 1e-6) / 1e12,
                   100.0 * (us / both_us - 1), worst_rel(got, want, count));
            free(got), free(want);
            cuMemFree(part);
        }
        cuMemFree(a), cuMemFree(as), cuMemFree(qb), cuMemFree(d), cuMemFree(c);
    }
    printf("  all three: %.1f TOPS\n", total_ops / (total_us * 1e-6) / 1e12);
    return 0;
}
