#include <iostream>
#include <omp.h>

using namespace std;

int main() {
    omp_set_num_threads(8);
    #pragma omp parallel num_threads(2)
    {
        int tid = omp_get_thread_num();
        printf("hello  %d\n", tid);
        // printf("hello again %d\n", tid);
    }
//    output:
//    hello  1
//    hello  0
//    hello  3
//    hello  2
//    hello again 1
//    hello again 0
//    hello again 3
//    hello again 2
    return 0;
}