// LD_PRELOAD shim: with CUPROF set, replays each graph launch as the same
// kernels launched one at a time, timed with events, and prints GPU time per
// kernel at exit. Assumes every kernel parameter is at most 8 bytes, which
// holds for phobos's exploded-memref ABI. Needs no CUDA headers.
//
//     gcc -O2 -shared -fPIC -o cuprof.so cuprof.c -ldl
//     CUPROF=1 LD_PRELOAD=./cuprof.so phobos-bench -m MODEL -p 16 -n 32 -r 1 --no-warmup
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    void *func;
    unsigned gx, gy, gz, bx, by, bz, shared;
    void **params;
    void **extra;
} KNP;

#define MAXP 512
typedef struct {
    void *node, *func;
    unsigned gx, gy, gz, bx, by, bz, shared;
    int np;
    unsigned long long vals[MAXP];
} Node;
typedef struct {
    void *graph, *exec;
    int n, cap;
    Node *nodes;
} Graph;

static Graph graphs[256];
static int ngraphs;
static int profile = -1;

typedef struct {
    const char *name;
    double ms;
    long n;
} Stat;
static Stat stats[512];
static int nstats;

static void *sym(const char *s) { return dlsym(RTLD_NEXT, s); }

static Graph *by_graph(void *g) {
    for (int i = 0; i < ngraphs; i++)
        if (graphs[i].graph == g) return &graphs[i];
    if (ngraphs == 256) return NULL;
    Graph *x = &graphs[ngraphs++];
    memset(x, 0, sizeof *x);
    x->graph = g;
    return x;
}
static Graph *by_exec(void *e) {
    for (int i = ngraphs - 1; i >= 0; i--)
        if (graphs[i].exec == e) return &graphs[i];
    return NULL;
}

static int param_count(void *func) {
    int (*info)(void *, size_t, size_t *, size_t *) = sym("cuFuncGetParamInfo");
    size_t off, size;
    int n = 0;
    while (n < MAXP && info(func, n, &off, &size) == 0) n++;
    return n;
}

static void fill(Node *nd, const KNP *p) {
    nd->func = p->func;
    nd->gx = p->gx, nd->gy = p->gy, nd->gz = p->gz;
    nd->bx = p->bx, nd->by = p->by, nd->bz = p->bz;
    nd->shared = p->shared;
    nd->np = param_count(p->func);
    for (int i = 0; i < nd->np; i++) {
        unsigned long long v = 0;
        memcpy(&v, p->params[i], 8);
        nd->vals[i] = v;
    }
}

int cuGraphAddKernelNode(void **node, void *graph, void *deps, size_t ndeps, const KNP *p) {
    static int (*real)(void **, void *, void *, size_t, const KNP *);
    if (!real) real = sym("cuGraphAddKernelNode");
    int r = real(node, graph, deps, ndeps, p);
    Graph *g = by_graph(graph);
    if (r == 0 && g) {
        if (g->n == g->cap) {
            g->cap = g->cap ? 2 * g->cap : 512;
            g->nodes = realloc(g->nodes, g->cap * sizeof(Node));
        }
        Node *nd = &g->nodes[g->n++];
        nd->node = *node;
        fill(nd, p);
    }
    return r;
}

int cuGraphInstantiate_v2(void **exec, void *graph, void *a, void *b, size_t c) {
    static int (*real)(void **, void *, void *, void *, size_t);
    if (!real) real = sym("cuGraphInstantiate_v2");
    int r = real(exec, graph, a, b, c);
    Graph *g = by_graph(graph);
    if (r == 0 && g) g->exec = *exec;
    return r;
}

int cuGraphExecKernelNodeSetParams(void *exec, void *node, const KNP *p) {
    static int (*real)(void *, void *, const KNP *);
    if (!real) real = sym("cuGraphExecKernelNodeSetParams");
    Graph *g = by_exec(exec);
    if (g)
        for (int i = 0; i < g->n; i++)
            if (g->nodes[i].node == node) fill(&g->nodes[i], p);
    return real(exec, node, p);
}

static void add(const char *name, double ms) {
    for (int i = 0; i < nstats; i++)
        if (!strcmp(stats[i].name, name)) {
            stats[i].ms += ms, stats[i].n++;
            return;
        }
    if (nstats < 512) stats[nstats++] = (Stat){name, ms, 1};
}

int cuGraphLaunch(void *exec, void *stream) {
    static int (*real)(void *, void *);
    if (!real) real = sym("cuGraphLaunch");
    if (profile < 0) profile = getenv("CUPROF") != NULL;
    Graph *g = by_exec(exec);
    if (!profile || !g) return real(exec, stream);
    int (*launch)(void *, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, void *, void **,
                  void **) = sym("cuLaunchKernel");
    int (*ev_create)(void **, unsigned) = sym("cuEventCreate");
    int (*ev_record)(void *, void *) = sym("cuEventRecord");
    int (*ev_sync)(void *) = sym("cuEventSynchronize");
    int (*ev_elapsed)(float *, void *, void *) = sym("cuEventElapsedTime");
    int (*fname)(const char **, void *) = sym("cuFuncGetName");
    static void *e0, *e1;
    if (!e0) ev_create(&e0, 0), ev_create(&e1, 0);
    for (int i = 0; i < g->n; i++) {
        Node *nd = &g->nodes[i];
        void *argv[MAXP];
        for (int k = 0; k < nd->np; k++) argv[k] = &nd->vals[k];
        ev_record(e0, stream);
        int r = launch(nd->func, nd->gx, nd->gy, nd->gz, nd->bx, nd->by, nd->bz, nd->shared, stream, argv, NULL);
        if (r) {
            const char *nm = "?";
            fname(&nm, nd->func);
            fprintf(stderr, "[cuprof] launch %d of %d failed: %s np=%d grid %u,%u,%u block %u shared %u r=%d\n", i,
                    g->n, nm, nd->np, nd->gx, nd->gy, nd->gz, nd->bx, nd->shared, r);
            return r;
        }
        ev_record(e1, stream);
        ev_sync(e1);
        float ms = 0;
        ev_elapsed(&ms, e0, e1);
        const char *name = "?";
        fname(&name, nd->func);
        add(name, ms);
    }
    return 0;
}

__attribute__((destructor)) static void report(void) {
    double total = 0;
    for (int i = 0; i < nstats; i++) total += stats[i].ms;
    for (int pass = 0; pass < nstats; pass++) {
        int best = -1;
        for (int i = 0; i < nstats; i++)
            if (stats[i].n >= 0 && (best < 0 || stats[i].ms > stats[best].ms)) best = i;
        if (best < 0) break;
        fprintf(stderr, "[cuprof] %-28s %8ld launches %10.2f ms %6.1f%% %9.1f us/launch\n", stats[best].name,
                stats[best].n, stats[best].ms, 100 * stats[best].ms / total, 1000 * stats[best].ms / stats[best].n);
        stats[best].n = -1;
    }
}
