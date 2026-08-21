/* False sharing — the classic multicore performance trap.

   Build:  gcc -O3 -fopenmp 05_false_sharing.c -o 05_false_sharing
   Run:    OMP_NUM_THREADS=48 ./05_false_sharing

   Each thread accumulates into its own slot of a shared results[] array.
   There is NO logical sharing — thread t only touches results[t] — so the
   answer is always correct. But cache coherence works on 64-byte lines,
   not individual doubles: 8 adjacent doubles live on ONE line. When thread
   0 writes results[0] and thread 1 writes results[1], both slots share a
   line, so every write invalidates the other core's copy. The line
   ping-pongs between cores (MESI), and the "parallel" loop crawls.

     (A) BAD  : contiguous double results[T]      -> false sharing.
     (B) GOOD : each thread accumulates in a LOCAL variable and writes its
                slot once at the very end          -> no line ping-pong.
     (C) GOOD : pad each slot to its own 64-byte cache line (struct+padding)
                -> adjacent slots never share a line.

   All three produce the same per-slot sums (checked vs a serial reference).
   Only the timing differs — that is the whole point.

   Note: the accumulators in (A) and (C) are 'volatile' on purpose. Without
   it, gcc -O3 would keep results[id] in a register for the whole loop and
   write it back only once — which would ELIMINATE the memory traffic and
   hide the very effect we want to show. 'volatile' forces a real
   load-modify-store to the cache line every iteration, so (A) actually
   suffers the coherence ping-pong and (C) actually benefits from padding.  */
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define ITERS (50L*1000*1000)   /* work per thread */

typedef struct { volatile double v; char pad[64 - sizeof(double)]; } Padded; /* one line each */

int main(void) {
    int T = omp_get_max_threads();

    /* serial reference: what each slot's sum must equal */
    double ref = 0.0;
    for (long k = 0; k < ITERS; k++) ref += 1.0;   /* = ITERS */

    /* (A) BAD: adjacent doubles, hammered in the inner loop -------------- */
    volatile double *bad = calloc(T, sizeof *bad);
    double t0 = omp_get_wtime();
    #pragma omp parallel
    {
        int id = omp_get_thread_num();
        for (long k = 0; k < ITERS; k++)
            bad[id] += 1.0;            /* neighbors share a cache line -> ping-pong */
    }
    double t_bad = omp_get_wtime() - t0;

    /* (B) GOOD: local accumulator, single write-back -------------------- */
    double *good = calloc(T, sizeof *good);
    t0 = omp_get_wtime();
    #pragma omp parallel
    {
        int id = omp_get_thread_num();
        double local = 0.0;           /* lives in a register/stack, private */
        for (long k = 0; k < ITERS; k++) local += 1.0;
        good[id] = local;             /* one shared write at the end */
    }
    double t_good = omp_get_wtime() - t0;

    /* (C) GOOD: cache-line padded slots --------------------------------- */
    Padded *pad = calloc(T, sizeof *pad);
    t0 = omp_get_wtime();
    #pragma omp parallel
    {
        int id = omp_get_thread_num();
        for (long k = 0; k < ITERS; k++)
            pad[id].v += 1.0;         /* each slot on its own line */
    }
    double t_pad = omp_get_wtime() - t0;

    /* correctness: every slot must equal the serial reference */
    int okA = 1, okB = 1, okC = 1;
    for (int i = 0; i < T; i++) {
        if (bad[i]  != ref) okA = 0;
        if (good[i] != ref) okB = 0;
        if (pad[i].v!= ref) okC = 0;
    }

    printf("threads (max)              : %d   (line = 64 B, %zu doubles/line)\n",
           T, (size_t)(64 / sizeof(double)));
    printf("(A) false sharing          : %.4f s   correctness %s\n",
           t_bad,  okA ? "PASS" : "FAIL");
    printf("(B) local accumulator      : %.4f s   correctness %s   speedup %.1fx vs (A)\n",
           t_good, okB ? "PASS" : "FAIL", t_bad / t_good);
    printf("(C) cache-line padded      : %.4f s   correctness %s   speedup %.1fx vs (A)\n",
           t_pad,  okC ? "PASS" : "FAIL", t_bad / t_pad);
    printf("inefficiency: (A) writes adjacent doubles on ONE cache line, so\n"
           "  cores fight over it (coherence ping-pong). Fix: give each thread\n"
           "  private storage (B) or pad each slot to its own line (C).\n");

    free((void *)bad); free(good); free(pad);
    return 0;
}

/* Reference run — 48-core Xeon w7-2495X, gcc -O3 -fopenmp, OMP_NUM_THREADS=48:

$ OMP_NUM_THREADS=48 ./05_false_sharing        # run 1
threads (max)              : 48   (line = 64 B, 8 doubles/line)
(A) false sharing          : 0.1843 s   correctness PASS
(B) local accumulator      : 0.0350 s   correctness PASS   speedup 5.3x vs (A)
(C) cache-line padded      : 0.1370 s   correctness PASS   speedup 1.3x vs (A)

$ OMP_NUM_THREADS=48 ./05_false_sharing        # run 2
(A) false sharing          : 0.1637 s   correctness PASS
(B) local accumulator      : 0.0314 s   correctness PASS   speedup 5.2x vs (A)
(C) cache-line padded      : 0.1384 s   correctness PASS   speedup 1.2x vs (A)

All three compute the same per-slot sums (PASS). (A) hammers adjacent
doubles that share one 64-byte line, so 48 cores fight over it (coherence
ping-pong) -> ~0.18 s. (B), a thread-private accumulator with a single
write-back, removes both the sharing AND the per-iteration memory traffic
-> ~5x faster: this is the real fix. (C) pads each slot onto its own line,
which removes the coherence ping-pong (that is why it beats (A)); it stays
slower than (B) here only because these accumulators are 'volatile' (see
top comment), so (C) still does a real DRAM store every iteration. Takeaway:
keep per-thread state thread-private, and when threads must share an array,
pad entries to a cache line.                                               */