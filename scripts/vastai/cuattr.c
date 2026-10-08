// LD_PRELOAD shim: prints the attributes the driver reports for every kernel
// it resolves, once per distinct (name, regs, local) combination.
//
//     gcc -O2 -shared -fPIC -o cuattr.so cuattr.c -ldl
//     LD_PRELOAD=./cuattr.so phobos-bench -m MODEL
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <string.h>

static char seen[256][160];
static int nseen;

int cuModuleGetFunction(void **func, void *module, const char *name) {
    static int (*real)(void **, void *, const char *);
    if (!real) real = dlsym(RTLD_NEXT, "cuModuleGetFunction");
    int r = real(func, module, name);
    if (r) return r;
    int (*attr)(int *, int, void *) = dlsym(RTLD_NEXT, "cuFuncGetAttribute");
    int threads = 0, shared = 0, local = 0, regs = 0, ptx = 0, bin = 0;
    attr(&threads, 0, *func);
    attr(&shared, 1, *func);
    attr(&local, 3, *func);
    attr(&regs, 4, *func);
    attr(&ptx, 5, *func);
    attr(&bin, 6, *func);
    char key[160];
    snprintf(key, sizeof key, "%s/%d/%d/%d", name, regs, local, threads);
    for (int i = 0; i < nseen; i++)
        if (!strcmp(seen[i], key)) return r;
    if (nseen < 256) strcpy(seen[nseen++], key);
    fprintf(stderr, "[cuattr] %-28s regs %3d local %5d B static shared %6d B max threads %4d ptx %d sass %d\n", name,
            regs, local, shared, threads, ptx, bin);
    return r;
}
