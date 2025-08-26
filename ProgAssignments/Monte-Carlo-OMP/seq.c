#include <stdlib.h>
#include "util.h"

double monte_carlo_pi(long long N, unsigned int seed) {
    long long hits = 0;
    srand(seed);

    for (long long i = 0; i < N; i++) {
        double x = (double)rand() / RAND_MAX;
        double y = (double)rand() / RAND_MAX;
        
        if (x * x + y * y <= 1.0) {
            hits++;
        }
    }

    double estimated_pi = 4.0 * (double)hits / (double)N;
    return estimated_pi;
}

double monte_carlo_integral(long long N, unsigned int seed) {
    double sum = 0.0;
    rng32_t rng = { .s = 0x85ebca6bu ^ (seed ? seed : 1u) };

    for(long long i=0; i<N; ++i) {
        double x = uniform_01(&rng);
        double y = f(x);
        sum  += y;
    }

    double I_hat = sum / (double)N;
    return I_hat;
}












/**************************** */









// // -------------------- RNGs (sequential, per-process) --------------------
// typedef struct { uint32_t s; } rng32_t;
// static inline uint32_t xorshift32(rng32_t* st){
//     uint32_t x = st->s; if(!x) x = 0x9e3779b9u;
//     x ^= x << 13; x ^= x >> 17; x ^= x << 5;
//     st->s = x; return x;
// }
// static inline float u01_xor(rng32_t* st){ // [0,1)
//     return (xorshift32(st) >> 8) * (1.0f/16777216.0f); // 24-bit mantissa
// }

// // Simple LCG as a stand-in; faster than mt19937 but worse quality.
// // (Pure C here to avoid C++ <random>; this is for the RNG microbenchmark demo only)
// typedef struct { uint64_t s; } lcg64_t;
// static inline uint64_t lcg64(lcg64_t* st){
//     st->s = st->s * 6364136223846793005ULL + 1ULL;
//     return st->s;
// }
// static inline float u01_lcg(lcg64_t* st){
//     return (float)((lcg64(st) >> 40) * (1.0/1099511627776.0)); // 24+ bits
// }

// // -------------------- Task 1: Monte Carlo pi --------------------
// typedef struct {
//     long long N;
//     unsigned seed;
// } PiArgs;

// static void task1_pi(const PiArgs* a){
//     rng32_t rng = { .s = 0x9e3779b9u ^ (a->seed ? a->seed : 1u) };
//     long long hits = 0;
//     double t0 = now_sec();
//     for(long long i=0;i<a->N;i++){
//         float x = u01_xor(&rng);
//         float y = u01_xor(&rng);
//         if(x*x + y*y <= 1.0f) hits++;
//     }
//     double t1 = now_sec();
//     double pi_hat = 4.0 * (double)hits / (double)a->N;
//     double abs_err = fabs(pi_hat - M_PI);
//     printf("task=pi, N=%lld, seed=%u, pi_hat=%.12f, abs_error=%.3e, time_sec=%.6f\n",
//            a->N, a->seed, pi_hat, abs_err, t1-t0);
// }

// // -------------------- Task 2: 1D Monte Carlo integration --------------------
// static inline double f1(double x){ return exp(-x*x); }
// static inline double f2(double x){ return sin(100.0*x)/(1.0+x); }

// typedef enum { FUNC_F1=1, FUNC_F2=2 } FuncKind;
// typedef enum { M_UNIFORM=1, M_STRAT=2, M_ANTITHETIC=3, M_IS=4 } MethodKind;

// typedef struct {
//     long long N;
//     unsigned seed;
//     FuncKind fkind;
//     MethodKind mkind;
//     int L; // strata count (for stratified)
// } IntArgs;

// // High-accuracy Simpson integration on [0,1] to get a "truth" reference.
// static double simpson_ref(double (*f)(double), int n){
//     if(n % 2) n++; // even
//     double h = 1.0 / n;
//     double s = f(0.0) + f(1.0);
//     for(int i=1;i<n;i++){
//         double x = i*h;
//         s += (i%2? 4.0: 2.0)*f(x);
//     }
//     return s * (h/3.0);
// }

// // Importance sampling helper: proposal p(x)=2x on [0,1]; CDF F(x)=x^2 => x = sqrt(u)
// static inline double sample_p_2x(double u){ return sqrt(u); }
// static inline double pdf_p_2x(double x){ return 2.0*x; } // x in [0,1]

// // Integrators (sequential)
// static void integrate_uniform(double (*f)(double), long long N, rng32_t* rng, double* est, double* var){
//     // Standard MC estimator: mean of f(U), U~U[0,1]
//     double s=0.0, ss=0.0;
//     for(long long i=0;i<N;i++){
//         double u = (double)u01_xor(rng);
//         double y = f(u);
//         s += y; ss += y*y;
//     }
//     double m = s/N;
//     double v = (ss/N - m*m); // population var
//     *est = m; *var = v;
// }

// static void integrate_stratified(double (*f)(double), long long N, int L, rng32_t* rng, double* est){
//     if(L <= 0) L = 1024;
//     // Divide [0,1] into L equal strata; sample approximately N/L per stratum
//     long long base = N / L, rem = N % L;
//     double acc = 0.0;
//     for(int k=0;k<L;k++){
//         long long nk = base + (k < rem ? 1 : 0);
//         if(nk == 0) continue;
//         double a = (double)k / L, b = (double)(k+1) / L;
//         double s = 0.0;
//         for(long long i=0;i<nk;i++){
//             double u = (double)u01_xor(rng);
//             double x = a + (b-a)*u;
//             s += f(x);
//         }
//         acc += (b-a) * (s/nk);
//     }
//     *est = acc;
// }

// static void integrate_antithetic(double (*f)(double), long long N, rng32_t* rng, double* est){
//     // Pair (u, 1-u); if N odd, one leftover plain sample
//     long long pairs = N/2;
//     double s = 0.0;
//     for(long long i=0;i<pairs;i++){
//         double u = (double)u01_xor(rng);
//         double a = f(u);
//         double b = f(1.0 - u);
//         s += 0.5*(a+b);
//     }
//     if(N%2){
//         double u = (double)u01_xor(rng);
//         s += f(u);
//         *est = (s)/(pairs + 1);
//     }else{
//         *est = (pairs? s/pairs : 0.0);
//     }
// }

// static void integrate_importance(double (*f)(double), long long N, rng32_t* rng, double* est){
//     // x ~ p(x)=2x, weight = f(x)/p(x)
//     double s = 0.0;
//     for(long long i=0;i<N;i++){
//         double u = (double)u01_xor(rng);
//         double x = sample_p_2x(u);
//         double w = f(x) / pdf_p_2x(x); // safe since x in (0,1], pdf>0 except x=0 (u=0 has prob 0)
//         s += w;
//     }
//     *est = s/N;
// }

// static void task2_integral(const IntArgs* a){
//     double (*f)(double) = (a->fkind==FUNC_F2) ? f2 : f1;

//     // Reference value via Simpson (dense); adjust n_ref for more accuracy
//     int n_ref = 200000; // 2e5 subintervals ~ pretty tight
//     double I_ref = simpson_ref(f, n_ref);

//     rng32_t rng = { .s = 0x85ebca6bu ^ (a->seed ? a->seed : 1u) };

//     double t0 = now_sec();
//     double est=0.0, var=0.0;

//     switch(a->mkind){
//         case M_UNIFORM:     integrate_uniform(f, a->N, &rng, &est, &var); break;
//         case M_STRAT:       integrate_stratified(f, a->N, a->L, &rng, &est); break;
//         case M_ANTITHETIC:  integrate_antithetic(f, a->N, &rng, &est); break;
//         case M_IS:          integrate_importance(f, a->N, &rng, &est); break;
//         default:            integrate_uniform(f, a->N, &rng, &est, &var); break;
//     }

//     double t1 = now_sec();
//     double abs_err = fabs(est - I_ref);
//     const char* fname = (a->fkind==FUNC_F2) ? "f2" : "f1";
//     const char* mname = (a->mkind==M_UNIFORM) ? "uniform" :
//                         (a->mkind==M_STRAT) ? "stratified" :
//                         (a->mkind==M_ANTITHETIC) ? "antithetic" :
//                         (a->mkind==M_IS) ? "is" : "uniform";

//     printf("task=int, func=%s, method=%s, N=%lld, seed=%u, I_hat=%.12f, I_ref=%.12f, abs_error=%.3e, time_sec=%.6f",
//            fname, mname, a->N, a->seed, est, I_ref, abs_err, t1-t0);
//     if(a->mkind==M_UNIFORM) printf(", var_est=%.6e", var);
//     if(a->mkind==M_STRAT) printf(", L=%d", a->L);
//     printf("\n");
// }

// // -------------------- Task 3 (one option): RNG microbenchmark --------------------
// typedef struct { long long N; unsigned seed; } RngArgs;
// static void task3_rng_bench(const RngArgs* a){
//     // Compare throughput (samples/sec) of xorshift32 vs LCG
//     const long long N = a->N;
//     volatile double sink = 0.0; // prevent optimization
//     double t0, t1;

//     rng32_t xr = { .s = 0x9e3779b9u ^ (a->seed? a->seed:1u) };
//     t0 = now_sec();
//     for(long long i=0;i<N;i++){
//         sink += u01_xor(&xr);
//     }
//     t1 = now_sec();
//     double t_xor = t1 - t0;

//     lcg64_t lr = { .s = 0x243F6A8885A308D3ULL ^ (uint64_t)(a->seed? a->seed:1u) };
//     t0 = now_sec();
//     for(long long i=0;i<N;i++){
//         sink += u01_lcg(&lr);
//     }
//     t1 = now_sec();
//     double t_lcg = t1 - t0;

//     printf("task=rng, N=%lld, seed=%u, xorshift32_time=%.6f s (%.2f M/s), lcg_time=%.6f s (%.2f M/s)\n",
//            N, a->seed,
//            t_xor, (N*1e-6)/t_xor,
//            t_lcg, (N*1e-6)/t_lcg);
//     // Use sink so compiler keeps loops
//     if(sink==12345.678) fprintf(stderr,"ignore: %f\n", sink);
// }

// // -------------------- CLI --------------------
// static void usage(const char* prog){
//     fprintf(stderr,
//       "Usage:\n"
//       "  %s -task pi  -n <samples> -seed <s>\n"
//       "  %s -task int -func <f1|f2> -method <uniform|stratified|antithetic|is> -n <samples> -seed <s> [-L <strata>]\n"
//       "  %s -task rng -n <samples> -seed <s>\n", prog, prog, prog);
// }

// int main(int argc, char** argv){
//     if(argc < 2){ usage(argv[0]); return 0; }

//     // Defaults
//     char task[16]="";
//     char func[8]="f1";
//     char method[16]="uniform";
//     long long N = 1000000;
//     unsigned seed = 42;
//     int L = 1024;

//     for(int i=1;i<argc;i++){
//         if(!strcmp(argv[i],"-task") && i+1<argc) strncpy(task, argv[++i], sizeof(task)-1);
//         else if(!strcmp(argv[i],"-func") && i+1<argc) strncpy(func, argv[++i], sizeof(func)-1);
//         else if(!strcmp(argv[i],"-method") && i+1<argc) strncpy(method, argv[++i], sizeof(method)-1);
//         else if(!strcmp(argv[i],"-n") && i+1<argc) N = atoll(argv[++i]);
//         else if(!strcmp(argv[i],"-seed") && i+1<argc) seed = (unsigned)strtoul(argv[++i], NULL, 10);
//         else if(!strcmp(argv[i],"-L") && i+1<argc) L = atoi(argv[++i]);
//         else if(!strcmp(argv[i],"-h") || !strcmp(argv[i],"--help")) { usage(argv[0]); return 0; }
//     }

//     if(!strcmp(task, "pi")){
//         PiArgs a = { .N=N, .seed=seed };
//         task1_pi(&a);
//     } else if(!strcmp(task, "int")){
//         FuncKind fk = (!strcmp(func,"f2"))? FUNC_F2 : FUNC_F1;
//         MethodKind mk = M_UNIFORM;
//         if(!strcmp(method,"uniform")) mk = M_UNIFORM;
//         else if(!strcmp(method,"stratified")) mk = M_STRAT;
//         else if(!strcmp(method,"antithetic")) mk = M_ANTITHETIC;
//         else if(!strcmp(method,"is")) mk = M_IS;

//         IntArgs a = { .N=N, .seed=seed, .fkind=fk, .mkind=mk, .L=L };
//         task2_integral(&a);
//     } else if(!strcmp(task, "rng")){
//         RngArgs a = { .N=N, .seed=seed };
//         task3_rng_bench(&a);
//     } else {
//         usage(argv[0]);
//     }
//     return 0;
// }
