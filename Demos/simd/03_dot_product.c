/* 03 - Dot product  s = sum(a[i]*b[i])  (CSC 548, Topic 5: SIMD)
 *
 * A REDUCTION: unlike vector-add/saxpy the output is one scalar, so the loop
 * carries a dependence through the accumulator. The SIMD lesson here is twofold:
 *   1. vectorize the multiply-accumulate with FMA across lanes, then do ONE
 *      horizontal sum at the end;
 *   2. break the single accumulator dependence chain by using SEVERAL
 *      independent accumulators, which hides FMA latency (~4 cycles) and lets
 *      the out-of-order core keep multiple FMAs in flight.
 *
 * Demonstrates, on a real Intel Xeon w7-2495X (AVX-512 capable):
 *   (a) CORRECTNESS  - double-precision scalar reference vs each SIMD kernel,
 *                      relative error + PASS/FAIL (float reduction reorders adds,
 *                      so we compare against a trusted double sum with tolerance).
 *   (b) SIZE SWEEP   - N = 1<<10, 1<<20, 1<<26.
 *   (c) INEFFICIENT vs OPTIMIZED, timed:
 *         - scalar reduction                                      [SLOW]
 *         - AVX2 FMA, ONE accumulator  (latency-bound)            [FASTER]
 *         - AVX2 FMA, FOUR accumulators (unrolled, latency-hidden)[FASTEST AVX2]
 *         - AVX-512 FMA, four accumulators (bonus).
 *       INEFFICIENCY: a single accumulator serializes on the FMA latency chain;
 *       FIX: multiple accumulators expose instruction-level parallelism.
 *
 * NUMERICS NOTE - vectorizing a reduction improves ACCURACY, not just speed:
 * this is a float32 sum. The naive SCALAR loop funnels all N products into one
 * float accumulator; at N=2^26 the running sum reaches ~1.3e7, where the float32
 * ULP is ~2, so each ~0.36-magnitude addend falls BELOW the ULP and is rounded
 * away entirely - the sum saturates and ends up ~26% low (relerr 2.6e-1). Each
 * SIMD kernel instead spreads the work over its lanes (and the 4-acc versions
 * over 32/64 partial sums), so every partial stays small and accurate. Watch
 * relerr shrink from scalar -> 1-acc -> 4-acc -> AVX-512. The SIMD kernels are
 * validated against a trusted DOUBLE reference within tol = 16*sqrt(N)*eps
 * (the expected O(sqrt(N)) growth of a well-split float32 reduction); the scalar
 * line is the inaccurate baseline that motivates all of the above.
 *
 * Build (AVX-512 enabled by -march=native on this CPU):
 *   gcc -O3 -march=native -mavx2 -mfma -o 03_dot_product 03_dot_product.c
 * AVX2-only baseline:
 *   gcc -O3 -mavx2 -mfma      -o 03_dot_product 03_dot_product.c
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <immintrin.h>
#include <time.h>

static double now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec + t.tv_nsec*1e-9; }

/* horizontal sum of 8 float lanes -> scalar. */
static float hsum256(__m256 v){
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 s  = _mm_add_ps(lo, hi);
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    return _mm_cvtss_f32(s);
}

/* INEFFICIENT scalar reduction. */
static float dot_scalar(const float *a, const float *b, int n){
    float s = 0.0f;
    for (int i = 0; i < n; i++) s += a[i]*b[i];
    return s;
}

/* AVX2 FMA, single accumulator: correct, but every FMA waits on the previous. */
static float dot_avx2_1acc(const float *a, const float *b, int n){
    __m256 acc = _mm256_setzero_ps();
    int i = 0;
    for (; i <= n - 8; i += 8)
        acc = _mm256_fmadd_ps(_mm256_load_ps(&a[i]), _mm256_load_ps(&b[i]), acc);
    float s = hsum256(acc);
    for (; i < n; i++) s += a[i]*b[i];
    return s;
}

/* OPTIMIZED AVX2 FMA, four independent accumulators (32 floats/iter). */
static float dot_avx2_4acc(const float *a, const float *b, int n){
    __m256 a0=_mm256_setzero_ps(),a1=_mm256_setzero_ps(),
           a2=_mm256_setzero_ps(),a3=_mm256_setzero_ps();
    int i = 0;
    for (; i <= n - 32; i += 32){
        a0 = _mm256_fmadd_ps(_mm256_load_ps(&a[i   ]), _mm256_load_ps(&b[i   ]), a0);
        a1 = _mm256_fmadd_ps(_mm256_load_ps(&a[i+ 8]), _mm256_load_ps(&b[i+ 8]), a1);
        a2 = _mm256_fmadd_ps(_mm256_load_ps(&a[i+16]), _mm256_load_ps(&b[i+16]), a2);
        a3 = _mm256_fmadd_ps(_mm256_load_ps(&a[i+24]), _mm256_load_ps(&b[i+24]), a3);
    }
    __m256 acc = _mm256_add_ps(_mm256_add_ps(a0,a1), _mm256_add_ps(a2,a3));
    float s = hsum256(acc);
    for (; i < n; i++) s += a[i]*b[i];
    return s;
}

#ifdef __AVX512F__
static float hsum512(__m512 v){ return _mm512_reduce_add_ps(v); }
/* BONUS: AVX-512 FMA, four accumulators (64 floats/iter). */
static float dot_avx512_4acc(const float *a, const float *b, int n){
    __m512 a0=_mm512_setzero_ps(),a1=_mm512_setzero_ps(),
           a2=_mm512_setzero_ps(),a3=_mm512_setzero_ps();
    int i = 0;
    for (; i <= n - 64; i += 64){
        a0 = _mm512_fmadd_ps(_mm512_load_ps(&a[i   ]), _mm512_load_ps(&b[i   ]), a0);
        a1 = _mm512_fmadd_ps(_mm512_load_ps(&a[i+16]), _mm512_load_ps(&b[i+16]), a1);
        a2 = _mm512_fmadd_ps(_mm512_load_ps(&a[i+32]), _mm512_load_ps(&b[i+32]), a2);
        a3 = _mm512_fmadd_ps(_mm512_load_ps(&a[i+48]), _mm512_load_ps(&b[i+48]), a3);
    }
    __m512 acc = _mm512_add_ps(_mm512_add_ps(a0,a1), _mm512_add_ps(a2,a3));
    float s = hsum512(acc);
    for (; i < n; i++) s += a[i]*b[i];
    return s;
}
#endif

/* trusted reference in double precision. */
static double dot_ref(const float *a, const float *b, int n){
    double s = 0.0;
    for (int i = 0; i < n; i++) s += (double)a[i]*(double)b[i];
    return s;
}

static double bench(float (*fn)(const float*,const float*,int),
                    const float *a, const float *b, int n, int reps, float *out){
    volatile float sink = fn(a,b,n);                 /* warm + keep result */
    double best = 1e30;
    for (int r = 0; r < reps; r++){
        double t0 = now(); sink = fn(a,b,n); double t1 = now();
        double ms = (t1-t0)*1e3; if (ms < best) best = ms;
    }
    *out = sink; return best;
}

static void run_size(int n){
    float *a = aligned_alloc(64, n*sizeof(float));
    float *b = aligned_alloc(64, n*sizeof(float));
    /* values in [~0,1) so the double reference stays well-conditioned. */
    for (int i = 0; i < n; i++){ a[i] = (float)((i%97)+1)/100.0f; b[i] = (float)((i%31)+1)/40.0f; }

    int reps = n < (1<<20) ? 2000 : (n < (1<<24) ? 50 : 5);
    double ref = dot_ref(a,b,n);
    /* tolerance for a float32 reduction: ~ O(sqrt(N))*eps (see NUMERICS NOTE). */
    double tol = 16.0 * sqrt((double)n) * FLT_EPSILON;

    float rs,r1,r4;
    double ts = bench(dot_scalar,     a,b,n,reps,&rs);
    double t1 = bench(dot_avx2_1acc,  a,b,n,reps,&r1);
    double t4 = bench(dot_avx2_4acc,  a,b,n,reps,&r4);
    double esc = fabs(rs-ref)/fabs(ref);                       /* scalar's own error */
    double e1 = fabs(r1-ref)/fabs(ref), e4 = fabs(r4-ref)/fabs(ref);
#ifdef __AVX512F__
    float r5; double t5 = bench(dot_avx512_4acc, a,b,n,reps,&r5);
    double e5 = fabs(r5-ref)/fabs(ref);
#endif

    double bytes = 2.0*n*sizeof(float);              /* read a,b */
    double gb = bytes/1e9;
    printf("N=2^%-2d (%8.2f MB touched)   ref=%.6g   tol=%.1e\n",
           (int)round(log2((double)n)), bytes/1e6, ref, tol);
    printf("  scalar        : %10.5f ms   %6.2f GB/s                 relerr=%.1e  %s\n",
           ts, gb/(ts/1e3), esc, esc<=tol?"baseline":"baseline (float32 saturates!)");
    printf("  AVX2 FMA x1acc: %10.5f ms   %6.2f GB/s   speedup x%.2f   relerr=%.1e  %s\n",
           t1, gb/(t1/1e3), ts/t1, e1, e1<=tol?"PASS":"FAIL");
    printf("  AVX2 FMA x4acc: %10.5f ms   %6.2f GB/s   speedup x%.2f   relerr=%.1e  %s\n",
           t4, gb/(t4/1e3), ts/t4, e4, e4<=tol?"PASS":"FAIL");
#ifdef __AVX512F__
    printf("  AVX512 x4acc  : %10.5f ms   %6.2f GB/s   speedup x%.2f   relerr=%.1e  %s\n",
           t5, gb/(t5/1e3), ts/t5, e5, e5<=tol?"PASS":"FAIL");
#endif
    free(a); free(b);
}

int main(void){
#ifdef __AVX512F__
    printf("[build: AVX-512 path active]\n\n");
#else
    printf("[build: AVX2 baseline only]\n\n");
#endif
    run_size(1<<10);
    run_size(1<<20);
    run_size(1<<26);
    return 0;
}

/* Reference run - Xeon w7-2495X (AVX-512), gcc 13.3, -O3 -march=native -mavx2 -mfma:
 *
 * [build: AVX-512 path active]
 *
 * N=2^10 (    0.01 MB touched)   ref=197.039   tol=6.1e-05
 *   scalar        :    0.00052 ms    15.68 GB/s                 relerr=8.9e-07  baseline
 *   AVX2 FMA x1acc:    0.00014 ms    58.25 GB/s   speedup x3.72   relerr=3.8e-08  PASS
 *   AVX2 FMA x4acc:    0.00005 ms   172.47 GB/s   speedup x11.00  relerr=3.8e-08  PASS
 *   AVX512 x4acc  :    0.00004 ms   183.25 GB/s   speedup x11.69  relerr=4.0e-08  PASS
 * N=2^20 (    8.39 MB touched)   ref=205518   tol=2.0e-03
 *   scalar        :    0.57571 ms    14.57 GB/s                 relerr=2.0e-04  baseline
 *   AVX2 FMA x1acc:    0.21265 ms    39.45 GB/s   speedup x2.71   relerr=1.7e-05  PASS
 *   AVX2 FMA x4acc:    0.21280 ms    39.42 GB/s   speedup x2.71   relerr=7.8e-07  PASS
 *   AVX512 x4acc  :    0.21858 ms    38.38 GB/s   speedup x2.63   relerr=2.2e-06  PASS
 * N=2^26 (  536.87 MB touched)   ref=1.31533e+07   tol=1.6e-02
 *   scalar        :   60.99581 ms     8.80 GB/s                 relerr=2.6e-01  baseline (float32 saturates!)
 *   AVX2 FMA x1acc:   30.23259 ms    17.76 GB/s   speedup x2.02   relerr=8.0e-03  PASS
 *   AVX2 FMA x4acc:   25.89509 ms    20.73 GB/s   speedup x2.36   relerr=6.4e-04  PASS
 *   AVX512 x4acc  :   25.73046 ms    20.87 GB/s   speedup x2.37   relerr=2.0e-04  PASS
 *
 * Reading it: TWO wins from vectorizing this reduction. (1) SPEED - the 4-accumulator
 * kernels run x2.4 over scalar at 2^26 (multiple accumulators hide the ~4-cycle FMA
 * latency so the core keeps several FMAs in flight; a single accumulator is capped
 * at x2.0). (2) ACCURACY - relerr falls scalar(2.6e-1) -> 1acc(8.0e-3) -> 4acc(6.4e-4)
 * -> AVX512(2.0e-4): the naive scalar float sum saturates once it dwarfs its addends
 * (see NUMERICS NOTE), while splitting across lanes/partials keeps every partial sum
 * small. All three SIMD kernels PASS the double-reference check within tol=16*sqrt(N)*eps;
 * the scalar line is the (correct-to-report) inaccurate baseline.
 */
