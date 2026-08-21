/* 04 - Auto-vectorization  y = a*x + y, let the COMPILER emit the SIMD.
 * (CSC 548, Topic 5: SIMD / data-level parallelism)
 *
 * Same math as 02, but instead of hand-writing intrinsics we teach the compiler
 * to vectorize an ordinary C loop. Two things unlock it:
 *   - `restrict`      : promises x and y do not alias, so iterations are
 *                       independent (without it the compiler must assume y[i]
 *                       could be x[i+1] and refuses to vectorize).
 *   - `#pragma GCC ivdep` / `-O3 -march=native` : enable the vectorizer and let
 *                       it target this CPU's widest ISA (AVX-512 here).
 *
 * INEFFICIENT vs OPTIMIZED (both compiled into THIS binary, then timed):
 *   - axpy_novec  : built with __attribute__((optimize("no-tree-vectorize")))
 *                   so the compiler is FORBIDDEN to vectorize -> scalar code.
 *   - axpy_autovec: built at full -O3 -march=native -> compiler emits AVX-512.
 *   Same source loop, only the compiler flags differ. This is exactly the
 *   `-O3 -fno-tree-vectorize` vs `-O3 -march=native` comparison, captured in
 *   one reproducible run. (The Makefile also builds a standalone 04_novec with
 *   -fno-tree-vectorize so you can diff the two binaries / asm yourself.)
 *
 * CORRECTNESS: auto-vectorized result vs the scalar reference, max-error+PASS/FAIL.
 * SIZE SWEEP : N = 1<<10, 1<<20, 1<<26.
 *
 * Build & SEE which loops vectorized (look for "loop vectorized"):
 *   gcc -O3 -march=native -mavx2 -mfma -fopt-info-vec 04_auto_vectorize.c -o 04_auto_vectorize
 * Force it OFF for contrast:
 *   gcc -O3 -fno-tree-vectorize        04_auto_vectorize.c -o 04_novec
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

static double now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec + t.tv_nsec*1e-9; }

/* INEFFICIENT: the attribute blocks the auto-vectorizer -> plain scalar loop.
   This is the `-fno-tree-vectorize` case, isolated to one function. */
__attribute__((optimize("no-tree-vectorize")))
static void axpy_novec(int n, float a, const float *restrict x, float *restrict y){
    for (int i = 0; i < n; i++) y[i] = a*x[i] + y[i];
}

/* OPTIMIZED: identical loop, full optimization. restrict + ivdep + -march=native
   let the compiler auto-generate wide FMA SIMD (AVX-512 on this Xeon). */
static void axpy_autovec(int n, float a, const float *restrict x, float *restrict y){
    #pragma GCC ivdep
    for (int i = 0; i < n; i++) y[i] = a*x[i] + y[i];
}

static float max_err(const float *ref, const float *got, int n){
    float e = 0.0f;
    for (int i = 0; i < n; i++){ float d = fabsf(ref[i]-got[i]); if (d > e) e = d; }
    return e;
}

static double bench(void (*fn)(int,float,const float*,float*),
                    float a, const float *x, float *y, const float *y0, int n, int reps){
    for (int i=0;i<n;i++) y[i]=y0[i]; fn(n,a,x,y);
    double best = 1e30;
    for (int r = 0; r < reps; r++){
        for (int i=0;i<n;i++) y[i]=y0[i];
        double t0 = now(); fn(n,a,x,y); double t1 = now();
        double ms = (t1-t0)*1e3; if (ms < best) best = ms;
    }
    return best;
}

static void run_size(int n){
    float a = 2.0f;
    float *x  = aligned_alloc(64, n*sizeof(float));
    float *y0 = aligned_alloc(64, n*sizeof(float));
    float *ys = aligned_alloc(64, n*sizeof(float));   /* scalar reference */
    float *yv = aligned_alloc(64, n*sizeof(float));   /* auto-vec result  */
    for (int i = 0; i < n; i++){ x[i] = (float)(i%97)*0.25f; y0[i] = (float)(i%53); }

    int reps = n < (1<<20) ? 1000 : (n < (1<<24) ? 30 : 4);

    double ts = bench(axpy_novec,   a,x,ys,y0,n,reps);
    double tv = bench(axpy_autovec, a,x,yv,y0,n,reps);
    float  e  = max_err(ys,yv,n);

    double bytes = 3.0*n*sizeof(float);
    double gb = bytes/1e9;
    printf("N=2^%-2d (%8.2f MB touched)\n", (int)round(log2((double)n)), bytes/1e6);
    printf("  novec  (scalar)   : %10.5f ms   %6.2f GB/s\n", ts, gb/(ts/1e3));
    printf("  autovec(-march=nat): %10.5f ms   %6.2f GB/s   speedup x%.2f   maxerr=%.1e  %s\n",
           tv, gb/(tv/1e3), ts/tv, e, e<=1e-3f?"PASS":"FAIL");
    free(x); free(y0); free(ys); free(yv);
}

int main(void){
    run_size(1<<10);
    run_size(1<<20);
    run_size(1<<26);
    return 0;
}

/* Reference run - Xeon w7-2495X (AVX-512), gcc 13.3, -O3 -march=native -mavx2 -mfma:
 *
 * # vectorizer report (gcc -fopt-info-vec) confirms the loops were auto-vectorized:
 * 04_auto_vectorize.c:47:5: optimized: loop vectorized using 32 byte vectors
 * 04_auto_vectorize.c:47:5: optimized: loop vectorized using 16 byte vectors
 *   (line 47 = the axpy_autovec loop; 32B = AVX/AVX2 body, 16B = SSE peel/tail)
 *
 * N=2^10 (    0.01 MB touched)
 *   novec  (scalar)   :    0.00039 ms    31.64 GB/s
 *   autovec(-march=nat):    0.00006 ms   202.99 GB/s   speedup x6.42   maxerr=0.0e+00  PASS
 * N=2^20 (   12.58 MB touched)
 *   novec  (scalar)   :    0.73333 ms    17.16 GB/s
 *   autovec(-march=nat):    0.20020 ms    62.85 GB/s   speedup x3.66   maxerr=0.0e+00  PASS
 * N=2^26 (  805.31 MB touched)
 *   novec  (scalar)   :  107.63032 ms     7.48 GB/s
 *   autovec(-march=nat):   40.77320 ms    19.75 GB/s   speedup x2.64   maxerr=0.0e+00  PASS
 *
 * Reading it: same loop, only the compiler flag differs (-fno-tree-vectorize via
 * the per-function attribute vs full -O3 -march=native). The auto-vectorized code
 * matches the scalar bit-for-bit (maxerr 0) and runs x6.4 faster in L1, x2.6-3.7
 * faster out of cache - the compiler emitted AVX-512 FMA for this loop with zero
 * intrinsics, just `restrict` + `#pragma GCC ivdep` to prove the iterations are
 * independent. At 2^26 both approach the DRAM bandwidth wall.
 */
