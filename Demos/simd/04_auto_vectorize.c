/* Auto-vectorization: let the compiler do it. The `restrict` keyword and
   `#pragma omp simd` tell it the iterations are independent.
   Build & SEE the report:
     gcc -O3 -march=native -fopenmp -fopt-info-vec 04_auto_vectorize.c -o 04_auto_vectorize
   (clang: -Rpass=loop-vectorize)                                          */
#include <stdio.h>
#include <stdlib.h>

void axpy(int n, float a, const float *restrict x, float *restrict y){
    #pragma omp simd
    for (int i = 0; i < n; i++) y[i] = a*x[i] + y[i];
}

int main(void){
    int n = 1 << 20;
    float *x = malloc(n*sizeof(float)), *y = malloc(n*sizeof(float));
    for (int i = 0; i < n; i++) { x[i] = 1.0f; y[i] = (float)i; }
    axpy(n, 2.0f, x, y);
    printf("y[10] = %.1f (expect 12.0)\n", y[10]);
    free(x); free(y);
    return 0;
}
