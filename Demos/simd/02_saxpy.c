/* SAXPY  y = a*x + y  with AVX + fused multiply-add, plus a scalar tail.
   Build: gcc -O3 -march=native -o 02_saxpy 02_saxpy.c                     */
#include <stdio.h>
#include <stdlib.h>
#include <immintrin.h>

int main(void){
    int n = 1 << 20;
    float a = 2.0f;
    float *x = aligned_alloc(32, n*sizeof(float));
    float *y = aligned_alloc(32, n*sizeof(float));
    for (int i = 0; i < n; i++) { x[i] = 1.0f; y[i] = (float)i; }

    __m256 va = _mm256_set1_ps(a);
    int i = 0;
    for (; i <= n - 8; i += 8) {
        __m256 vx = _mm256_load_ps(&x[i]);
        __m256 vy = _mm256_load_ps(&y[i]);
        _mm256_store_ps(&y[i], _mm256_fmadd_ps(va, vx, vy));   /* y = a*x + y */
    }
    for (; i < n; i++) y[i] = a*x[i] + y[i];                    /* remainder */

    printf("y[10] = %.1f (expect %.1f)\n", y[10], a*1.0f + 10);
    free(x); free(y);
    return 0;
}
