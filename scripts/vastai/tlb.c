// Read-bandwidth sweep over access patterns and allocation kinds, to tell a
// page-translation bottleneck from an access-pattern one.
//
// A warp reads R rows `stride` bytes apart, 32/R lanes per row, 16 bytes per
// lane per step, walking each row to its end; warps stride over row groups
// until the buffer has been read once. R=1 is the Q8_0 matvec's shape (lanes
// along k); R=8 at a 1440 or 5184-byte stride is the K-quant matvec's.
//
//     gcc -O2 -o tlb tlb.c -I. -lcuda && ./tlb [GiB]
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

static const char *PTX =
    ".version 8.0\n"
    ".target sm_75\n"
    ".address_size 64\n"
    ".visible .entry sweep(.param .u64 p_base, .param .u64 p_rows, .param .u32 p_stride,\n"
    "                      .param .u32 p_r, .param .u32 p_shift, .param .u64 p_out)\n"
    "{\n"
    "  .reg .pred %p<4>;\n"
    "  .reg .b32 %r<32>;\n"
    "  .reg .b64 %rd<16>;\n"
    "  ld.param.u64 %rd1, [p_base];\n"
    "  ld.param.u64 %rd2, [p_rows];\n"
    "  ld.param.u32 %r1, [p_stride];\n"
    "  ld.param.u32 %r2, [p_r];\n"
    "  ld.param.u32 %r3, [p_shift];\n"
    "  mov.u32 %r4, %tid.x;\n"
    "  and.b32 %r5, %r4, 31;\n"          // lane
    "  shr.u32 %r6, %r4, 5;\n"           // warp in block
    "  mov.u32 %r7, %ntid.x;\n"
    "  shr.u32 %r7, %r7, 5;\n"           // warps per block
    "  mov.u32 %r8, %ctaid.x;\n"
    "  mad.lo.u32 %r9, %r8, %r7, %r6;\n" // global warp g
    "  mov.u32 %r10, %nctaid.x;\n"
    "  mul.lo.u32 %r10, %r10, %r7;\n"    // total warps W
    "  shr.u32 %r11, %r5, %r3;\n"        // row within group
    "  mov.u32 %r12, 1;\n"
    "  shl.b32 %r12, %r12, %r3;\n"       // lanes per row
    "  sub.u32 %r13, %r12, 1;\n"
    "  and.b32 %r14, %r5, %r13;\n"       // chunk index within row
    "  shl.b32 %r15, %r12, 4;\n"         // bytes per step
    "  mov.u32 %r20, 0;\n"               // accumulator
    "GROUP:\n"
    "  mad.lo.u32 %r16, %r9, %r2, %r11;\n"
    "  cvt.u64.u32 %rd3, %r16;\n"
    "  setp.ge.u64 %p1, %rd3, %rd2;\n"
    "  @%p1 bra DONE;\n"
    "  cvt.u64.u32 %rd4, %r1;\n"
    "  mul.lo.u64 %rd5, %rd3, %rd4;\n"
    "  add.u64 %rd5, %rd5, %rd1;\n"      // row address
    "  shl.b32 %r17, %r14, 4;\n"         // offset within row
    "STEP:\n"
    "  setp.ge.u32 %p2, %r17, %r1;\n"
    "  @%p2 bra NEXT;\n"
    "  cvt.u64.u32 %rd6, %r17;\n"
    "  add.u64 %rd7, %rd5, %rd6;\n"
    "  ld.global.v4.u32 {%r21, %r22, %r23, %r24}, [%rd7];\n"
    "  xor.b32 %r20, %r20, %r21;\n"
    "  xor.b32 %r20, %r20, %r22;\n"
    "  xor.b32 %r20, %r20, %r23;\n"
    "  xor.b32 %r20, %r20, %r24;\n"
    "  add.u32 %r17, %r17, %r15;\n"
    "  bra STEP;\n"
    "NEXT:\n"
    "  add.u32 %r9, %r9, %r10;\n"
    "  bra GROUP;\n"
    "DONE:\n"
    "  setp.ne.u32 %p3, %r20, 305419896;\n"
    "  @%p3 bra END;\n"
    "  ld.param.u64 %rd8, [p_out];\n"
    "  st.global.u32 [%rd8], %r20;\n"
    "END:\n"
    "  ret;\n"
    "}\n";

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

static CUfunction fn;
static CUevent e0, e1;

static double run(CUdeviceptr base, size_t bytes, unsigned stride, unsigned r, unsigned blocks, CUdeviceptr out) {
    unsigned long long rows = bytes / stride;
    rows -= rows % r;
    unsigned shift = 0;  // log2(32 / r), the lanes per row
    while ((32u >> shift) > r) shift++;
    void *args[] = {&base, &rows, &stride, &r, &shift, &out};
    CK(cuLaunchKernel(fn, blocks, 1, 1, 256, 1, 1, 0, 0, args, NULL));  // warmup
    CK(cuEventRecord(e0, 0));
    const int reps = 5;
    for (int i = 0; i < reps; i++) CK(cuLaunchKernel(fn, blocks, 1, 1, 256, 1, 1, 0, 0, args, NULL));
    CK(cuEventRecord(e1, 0));
    CK(cuEventSynchronize(e1));
    float ms = 0;
    CK(cuEventElapsedTime(&ms, e0, e1));
    return (double)rows * stride * reps / (ms * 1e-3) / 1e9;
}

int main(int argc, char **argv) {
    double gib = argc > 1 ? atof(argv[1]) : 2.5;
    CK(cuInit(0));
    CUdevice dev;
    CK(cuDeviceGet(&dev, 0));
    CUcontext ctx;
    CK(cuDevicePrimaryCtxRetain(&ctx, dev));
    CK(cuCtxSetCurrent(ctx));
    char name[128];
    CK(cuDeviceGetName(name, sizeof name, dev));
    int sms, vmm;
    CK(cuDeviceGetAttribute(&sms, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev));
    CK(cuDeviceGetAttribute(&vmm, CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED, dev));
    printf("%s, %d SMs, VMM supported %d\n", name, sms, vmm);

    CUmodule mod;
    CK(cuModuleLoadData(&mod, PTX));
    CK(cuModuleGetFunction(&fn, mod, "sweep"));
    CK(cuEventCreate(&e0, 0));
    CK(cuEventCreate(&e1, 0));
    CUdeviceptr out;
    CK(cuMemAlloc(&out, 4));

    size_t bytes = (size_t)(gib * (1ull << 30));
    bytes -= bytes % (2u << 20);

    // A: one cuMemAlloc, as the arena's slabs are.
    CUdeviceptr a;
    CK(cuMemAlloc(&a, bytes));
    CK(cuMemsetD32(a, 0x01020304, bytes / 4));

    // B: VMM at the recommended granularity.
    CUmemAllocationProp prop = {0};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = dev;
    size_t gmin = 0, grec = 0;
    CUdeviceptr b = 0;
    if (vmm) {
        CK(cuMemGetAllocationGranularity(&gmin, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
        CK(cuMemGetAllocationGranularity(&grec, &prop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED));
        printf("VMM granularity: minimum %zu, recommended %zu\n", gmin, grec);
        size_t vbytes = (bytes + grec - 1) / grec * grec;
        CUmemGenericAllocationHandle h;
        CK(cuMemCreate(&h, vbytes, &prop, 0));
        CK(cuMemAddressReserve(&b, vbytes, 2u << 20, 0, 0));
        CK(cuMemMap(b, vbytes, 0, h, 0));
        CUmemAccessDesc acc = {0};
        acc.location = prop.location;
        acc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        CK(cuMemSetAccess(b, vbytes, &acc, 1));
        CK(cuMemsetD32(b, 0x01020304, bytes / 4));
    }
    printf("buffer %.2f GiB; GB/s per pattern\n", bytes / (double)(1ull << 30));

    struct {
        const char *what;
        unsigned stride, r;
    } pats[] = {
        {"contiguous rows, 1/warp (Q8 shape)", 1440, 1},
        {"K-quant shape, 8 rows/warp @1440", 1440, 8},
        {"K-quant shape, 8 rows/warp @5184", 5184, 8},
        {"8 rows/warp @4096", 4096, 8},
        {"8 rows/warp @65536", 65536, 8},
        {"8 rows/warp @2MiB", 2u << 20, 8},
        {"32 rows/warp @4096", 4096, 32},
        {"32 rows/warp @65536", 65536, 32},
    };
    unsigned grids[] = {56, 4 * (unsigned)sms, 64 * (unsigned)sms};
    printf("%-38s %6s %10s %10s\n", "pattern", "blocks", "cuMemAlloc", "VMM");
    for (unsigned p = 0; p < sizeof pats / sizeof *pats; p++)
        for (unsigned g = 0; g < 3; g++) {
            double ga = run(a, bytes, pats[p].stride, pats[p].r, grids[g], out);
            double gb = vmm ? run(b, bytes, pats[p].stride, pats[p].r, grids[g], out) : 0;
            printf("%-38s %6u %10.1f %10.1f\n", pats[p].what, grids[g], ga, gb);
            fflush(stdout);
        }
    return 0;
}
