// Int8 tensor-core throughput of mma.sync m8n8k16 against m16n8k32, from
// registers only: eight independent accumulator chains per warp, no memory
// traffic. Tells whether a chip runs the Turing shape at its full rate.
//
//     gcc -O2 -o tc tc.c -I. -lcuda && ./tc
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

#define CHAINS 8

// One kernel per shape. Each of CHAINS accumulators takes ITERS mma in
// sequence; the operands are lane-dependent so nothing folds away.
static void kernel(char *out, const char *name, int big) {
    int dregs = big ? 4 : 2, aregs = big ? 4 : 1, bregs = big ? 2 : 1;
    char *p = out;
    p += sprintf(p,
                 ".version 8.0\n.target sm_80\n.address_size 64\n"
                 ".visible .entry %s(.param .u32 iters, .param .u64 outp)\n{\n"
                 "  .reg .pred %%p<2>;\n  .reg .b32 %%a<4>;\n  .reg .b32 %%b<2>;\n  .reg .b32 %%d<%d>;\n"
                 "  .reg .b32 %%r<8>;\n  .reg .b64 %%rd<2>;\n"
                 "  ld.param.u32 %%r0, [iters];\n  mov.u32 %%r1, %%tid.x;\n",
                 name, CHAINS * dregs);
    for (int i = 0; i < aregs; i++) p += sprintf(p, "  add.u32 %%a%d, %%r1, %d;\n", i, 0x01010101 * (i + 1));
    for (int i = 0; i < bregs; i++) p += sprintf(p, "  xor.b32 %%b%d, %%r1, %d;\n", i, 0x02020202 * (i + 1));
    for (int i = 0; i < CHAINS * dregs; i++) p += sprintf(p, "  mov.u32 %%d%d, 0;\n", i);
    p += sprintf(p, "LOOP:\n");
    for (int c = 0; c < CHAINS; c++) {
        int d0 = c * dregs;
        if (big)
            p += sprintf(p,
                         "  mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%%d%d,%%d%d,%%d%d,%%d%d}, "
                         "{%%a0,%%a1,%%a2,%%a3}, {%%b0,%%b1}, {%%d%d,%%d%d,%%d%d,%%d%d};\n",
                         d0, d0 + 1, d0 + 2, d0 + 3, d0, d0 + 1, d0 + 2, d0 + 3);
        else
            p += sprintf(p,
                         "  mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%%d%d,%%d%d}, {%%a0}, {%%b0}, "
                         "{%%d%d,%%d%d};\n",
                         d0, d0 + 1, d0, d0 + 1);
    }
    p += sprintf(p, "  sub.u32 %%r0, %%r0, 1;\n  setp.ne.u32 %%p0, %%r0, 0;\n  @%%p0 bra LOOP;\n");
    p += sprintf(p, "  mov.u32 %%r2, 0;\n");
    for (int i = 0; i < CHAINS * dregs; i++) p += sprintf(p, "  add.u32 %%r2, %%r2, %%d%d;\n", i);
    p += sprintf(p,
                 "  setp.ne.u32 %%p1, %%r2, 123456789;\n  @%%p1 bra END;\n"
                 "  ld.param.u64 %%rd1, [outp];\n  st.global.u32 [%%rd1], %%r2;\n"
                 "END:\n  ret;\n}\n");
}

int main(void) {
    CK(cuInit(0));
    CUdevice dev;
    CK(cuDeviceGet(&dev, 0));
    CUcontext ctx;
    CK(cuDevicePrimaryCtxRetain(&ctx, dev));
    CK(cuCtxSetCurrent(ctx));
    char name[128];
    int sms, clock_khz;
    CK(cuDeviceGetName(name, sizeof name, dev));
    CK(cuDeviceGetAttribute(&sms, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev));
    CK(cuDeviceGetAttribute(&clock_khz, CU_DEVICE_ATTRIBUTE_CLOCK_RATE, dev));
    printf("%s, %d SMs, %d MHz rated\n", name, sms, clock_khz / 1000);
    CUdeviceptr out;
    CK(cuMemAlloc(&out, 4));
    CUevent e0, e1;
    CK(cuEventCreate(&e0, 0));
    CK(cuEventCreate(&e1, 0));
    static char src[1 << 16];
    struct {
        const char *name;
        int big;
        double macs;
    } shapes[] = {{"m8n8k16", 0, 8 * 8 * 16}, {"m16n8k32", 1, 16 * 8 * 32}};
    for (int s = 0; s < 2; s++) {
        kernel(src, shapes[s].name, shapes[s].big);
        CUmodule mod;
        CK(cuModuleLoadData(&mod, src));
        CUfunction fn;
        CK(cuModuleGetFunction(&fn, mod, shapes[s].name));
        for (int warps = 4; warps <= 16; warps *= 2) {
            unsigned iters = 4096, blocks = sms * 4, threads = warps * 32;
            void *args[] = {&iters, &out};
            CK(cuLaunchKernel(fn, blocks, 1, 1, threads, 1, 1, 0, 0, args, NULL));
            CK(cuCtxSynchronize());
            CK(cuEventRecord(e0, 0));
            CK(cuLaunchKernel(fn, blocks, 1, 1, threads, 1, 1, 0, 0, args, NULL));
            CK(cuEventRecord(e1, 0));
            CK(cuEventSynchronize(e1));
            float ms;
            CK(cuEventElapsedTime(&ms, e0, e1));
            double ops = 2.0 * shapes[s].macs * CHAINS * iters * (double)blocks * warps;
            printf("  %-9s %2d warps/CTA, %4u CTAs: %8.1f TOPS\n", shapes[s].name, warps, blocks,
                   ops / (ms * 1e-3) / 1e12);
        }
    }
    return 0;
}
