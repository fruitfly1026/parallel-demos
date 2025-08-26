#include <stdlib.h>
#include <omp.h>

#include "util.h"

double monte_carlo_pi_omp(long long N, unsigned int seed) {
    long long hits = 0;

    /* TODO: Put your OpenMP code of monte_carlo_pi here. 
    Note: You need to generate seed per thread. */

    double estimated_pi = 0.0; // TODO: change this line
    return estimated_pi;
}

double monte_carlo_integral_omp(long long N, unsigned int seed) {
    double sum = 0.0;

    /* TODO: Put your OpenMP code of monte_carlo_integral here. 
    Note: You need to generate seed per thread, you can call "mix_seed" function defined in "util.h" file. */

    double I_hat = 0.0; // TODO: change this line
    return I_hat;
}