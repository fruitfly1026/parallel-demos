#include <stdlib.h>
#include <omp.h>

#include "util.h"

double monte_carlo_pi_omp(long long N, unsigned int seed) {
    long long hits = 0;

    /* TODO: Put your OpenMP code of monte_carlo_pi here */
    #pragma omp parallel reduction(+:hits)
    {
        unsigned int local_seed = seed ^ omp_get_thread_num();

        #pragma omp for schedule(static)
        for (long long i = 0; i < N; i++) {
            double x = (double)rand_r(&local_seed) / RAND_MAX;
            double y = (double)rand_r(&local_seed) / RAND_MAX;

            if (x * x + y * y <= 1.0) {
                hits++;
            }
        }
    }

    double estimated_pi = 4.0 * (double)hits / (double)N; // TODO: change this line
    return estimated_pi;
}

double monte_carlo_integral_omp(long long N, unsigned int seed) {
    double sum = 0.0;

    /* TODO: Put your OpenMP code of monte_carlo_integral here */
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        rng32_t rng = { .s = mix_seed(seed ? seed : 1u, (uint32_t)tid) };

        #pragma omp for schedule(static) reduction(+:sum)
        for(long long i = 0; i < N; ++i){
            double x = uniform_01(&rng);
            double y = f(x);
            sum += y;
        }
    }

    double I_hat = sum / (double)N; // TODO: change this line
    return I_hat;
}