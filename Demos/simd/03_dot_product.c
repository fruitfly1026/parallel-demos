/* Dot product with AVX FMA + a horizontal sum to reduce the 8 lanes.
   Build: gcc -O3 -march=native -o 03_dot_product 03_dot_product.c         */
#include <stdio.h>
#include <stdlib.h>
#include <immintrin.h>

static float hsum256(__m256 v){
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 s  = _mm_add_ps(lo, hi);
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    return _mm_cvtss_f32(s);
}

int main(void){
    int n = 1 << 20;
    float *a = aligned_alloc(32, n*sizeof(float));
    float *b = aligned_alloc(32, n*sizeof(float));
    for (int i = 0; i < n; i++) { a[i] = 1.0f; b[i] = 2.0f; }

    __m256 acc = _mm256_setzero_ps();
    int i = 0;
    for (; i <= n - 8; i += 8)
        acc = _mm256_fmadd_ps(_mm256_load_ps(&a[i]), _mm256_load_ps(&b[i]), acc);
    float dot = hsum256(acc);
    for (; i < n; i++) dot += a[i]*b[i];

    printf("dot = %.1f (expect %.1f)\n", dot, 2.0f*n);
    free(a); free(b);
    return 0;
}
