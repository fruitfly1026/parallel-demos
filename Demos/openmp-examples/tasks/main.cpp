#include <iostream>
#include <omp.h>

using namespace std;

int comp_fib_numbers (int n) {
    int fn1, fn2;
    if ( n == 0 || n == 1 ) return(n);
    #pragma omp task shared(fn1)
    fn1 = comp_fib_numbers(n-1);
    #pragma omp task shared(fn2)
    fn2 = comp_fib_numbers(n-2);
    #pragma omp taskwait
    return(fn1 + fn2);
}

int main() {
    // omp_set_num_threads(8);
    int n = 40;
    int fn = 0;
    fn = comp_fib_numbers(n);
    cout << "Fibonacci number " << n << " is " << fn << endl;
    // Fibonacci number 40 is 102334155

    return 0;
}