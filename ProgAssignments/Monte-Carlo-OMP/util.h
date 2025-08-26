#include <time.h>
#include <stdint.h>
#include <math.h>

#ifndef UTIL_H
#define UTIL_H

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

typedef struct { uint32_t s; } rng32_t;

static inline double f(double x) { return exp(-x*x); }

static inline uint32_t xorshift32(rng32_t* st){
    uint32_t x = st->s ? st->s : 0x9e3779b9u;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    st->s = x; return x;
}

static inline double uniform_01(rng32_t* st) {
    // use top 24 bits -> double in [0,1)
    return (xorshift32(st) >> 8) * (1.0/16777216.0); // 2^24
}

static inline uint32_t mix_seed(uint32_t seed, uint32_t tid){
    uint32_t x = seed ^ (0x9e3779b9u * (tid + 1));
    x ^= x >> 16; x *= 0x85ebca6bu;
    x ^= x >> 13; x *= 0xc2b2ae35u;
    x ^= x >> 16;
    return x ? x : 1u;
}

// -------------------- Timing --------------------
// static double now_sec(void){
// #if defined(CLOCK_MONOTONIC)
//     struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
//     return ts.tv_sec + ts.tv_nsec*1e-9;
// #else
//     struct timeval tv; gettimeofday(&tv, NULL);
//     return tv.tv_sec + tv.tv_usec*1e-6;
// #endif
// }

#endif // UTIL_H