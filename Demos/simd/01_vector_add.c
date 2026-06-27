/* Scalar vs AVX vector add of two float arrays (x86, AVX/AVX2).
   Build: gcc -O3 -march=native -o 01_vector_add 01_vector_add.c          */
#include <stdio.h>
#include <stdlib.h>
#include <immintrin.h>
#include <sys/time.h>

static double now(void){ struct timeval t; gettimeofday(&t,NULL); return t.tv_sec + t.tv_usec*1e-6; }

int main(void){
    int n = 1 << 24;                 /* 16M, divisible by 8 */
    float *a = aligned_alloc(32, n*sizeof(float));
    float *b = aligned_alloc(32, n*sizeof(float));
    float *c = aligned_alloc(32, n*sizeof(float));
    for (int i = 0; i < n; i++) { a[i] = i*0.5f; b[i] = i*2.0f; }

    double t0 = now();
    for (int i = 0; i < n; i++) c[i] = a[i] + b[i];          /* scalar */
    double t1 = now();
    for (int i = 0; i < n; i += 8) {                          /* AVX: 8 floats/iter */
        __m256 va = _mm256_load_ps(&a[i]);
        __m256 vb = _mm256_load_ps(&b[i]);
        _mm256_store_ps(&c[i], _mm256_add_ps(va, vb));
    }
    double t2 = now();

    printf("scalar: %7.2f ms\n", (t1-t0)*1e3);
    printf("AVX   : %7.2f ms\n", (t2-t1)*1e3);
    printf("check : c[1000] = %.1f (expect %.1f)\n", c[1000], a[1000]+b[1000]);
    free(a); free(b); free(c);
    return 0;
}
