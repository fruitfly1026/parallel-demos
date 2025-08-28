#include <iostream>
#include <omp.h>

int main() {
    
    int shared_sum = 0;
    // omp_set_num_threads(3);
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int private_sum=0;
        private_sum++;
        // #pragma omp atomic
        //     shared_sum++;    // not safe
        // printf("[%d:] private sum is %d\n", tid, private_sum);
        #pragma omp atomic
            shared_sum++;   // not safe
        printf("[%d:] shared sum is %d\n", tid, shared_sum);
    }
    std::cout << "final shared sum is " << shared_sum << std::endl;

//    output:
//    private sum is 1
//    private sum is 1
//    private sum is 1
//    shared sum is 3
//    shared sum is 3
//    shared sum is 3
//    final shared sum is 3
    return 0;
}