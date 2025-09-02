#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#include "util.h"

static const double I_true = 0.7468241328124271;

// double monte_carlo_pi(long long N, unsigned int seed);
double monte_carlo_pi_omp(long long N, unsigned int seed);
// double monte_carlo_integral(long long N, unsigned int seed);
double monte_carlo_integral_omp(long long N, unsigned int seed);

int main(int argc, char* argv[]) {
    long long N = 1000000;
    unsigned int seed = (unsigned int)time(NULL);

    if (argc > 1) {
        N = atoll(argv[1]);
    }
    else if (argc > 2) {
        seed = (unsigned int)atoi(argv[2]);
    } else {
        printf("Usage: %s <num_samples> <seed>\n", argv[0]);
        return -1;
    }

    // Run Task 1: Monte Carlo Pi Estimation
    struct timeval start;
    struct timeval end;

    gettimeofday(&start, NULL);
    double pi_hat = monte_carlo_pi_omp(N, seed);
    gettimeofday(&end, NULL);

    double abs_err = fabs(pi_hat - M_PI);
    double elapsed = (double)(end.tv_sec - start.tv_sec) +  (double)(end.tv_usec - start.tv_usec)/1000000.0f;

    printf("Samples N = %lld, Seed = %u\n", N, seed);
    printf("===== Task 1: Monte Carlo Pi Estimation =====\n");
    printf("Estimated pi = %.12f, True pi = %.12f\n", pi_hat, M_PI);
    printf("Absolute error = %.6e\n", abs_err);
    printf("Time elapsed = %.6f sec\n", elapsed);

    // Run Task 2: Monte Carlo Integration
    gettimeofday(&start, NULL);
    double I_hat = monte_carlo_integral_omp(N, seed);
    gettimeofday(&end, NULL);
    abs_err = fabs(I_hat - I_true);
    elapsed = (double)(end.tv_sec - start.tv_sec) +  (double)(end.tv_usec - start.tv_usec)/1000000.0f;
    printf("===== Task 2: Monte Carlo Integration =====\n");
    printf("Estimated I = %.12f, True I = %.12f\n", I_hat, I_true);
    printf("Absolute error = %.6e\n", abs_err);
    printf("Time elapsed = %.6f sec\n", elapsed);

    return 0;
}
