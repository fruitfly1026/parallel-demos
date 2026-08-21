/* Hello, OpenMP — a parallel region; every thread prints its id.
   The team size is set by OMP_NUM_THREADS at run time (no recompile).

   Build:  gcc -O3 -fopenmp 01_hello.c -o 01_hello
   Run:    OMP_NUM_THREADS=1  ./01_hello
           OMP_NUM_THREADS=4  ./01_hello
           OMP_NUM_THREADS=48 ./01_hello

   Concept: outside a parallel region there is one thread; inside
   '#pragma omp parallel' the runtime forks a TEAM of threads that run
   the block concurrently, then joins.  omp_get_max_threads() is what a
   parallel region WOULD use; omp_get_num_threads() is the actual team
   size, which is 1 in serial code and >1 only inside the region.        */
#include <omp.h>
#include <stdio.h>

int main(void) {
    /* Serial context: exactly one thread is running right now. */
    printf("serial region : omp_get_num_threads() = %d, "
           "omp_get_max_threads() = %d\n",
           omp_get_num_threads(), omp_get_max_threads());

    int team = 0;
    #pragma omp parallel
    {
        int id  = omp_get_thread_num();     /* 0 .. team-1  */
        int n   = omp_get_num_threads();     /* actual team size */
        #pragma omp master
        team = n;                            /* record team size once */
        printf("  hello from thread %2d of %2d\n", id, n);
    }

    printf("joined back to serial : team size was %d\n", team);
    return 0;
}

/* Reference run — 48-core Xeon w7-2495X, gcc -O3 -fopenmp:

$ OMP_NUM_THREADS=1 ./01_hello
serial region : omp_get_num_threads() = 1, omp_get_max_threads() = 1
  hello from thread  0 of  1
joined back to serial : team size was 1

$ OMP_NUM_THREADS=4 ./01_hello
serial region : omp_get_num_threads() = 1, omp_get_max_threads() = 4
  hello from thread  0 of  4
  hello from thread  3 of  4
  hello from thread  2 of  4
  hello from thread  1 of  4
joined back to serial : team size was 4

$ OMP_NUM_THREADS=48 ./01_hello
serial region : omp_get_num_threads() = 1, omp_get_max_threads() = 48
  hello from thread 26 of 48
  hello from thread 44 of 48
  hello from thread 25 of 48
  ... (48 lines total, order varies run to run) ...
  hello from thread 13 of 48
joined back to serial : team size was 48

Note: max_threads reflects OMP_NUM_THREADS even in serial code, while
num_threads is 1 until the parallel region opens. Print order inside the
region is nondeterministic — threads run concurrently and race stdout.  */
