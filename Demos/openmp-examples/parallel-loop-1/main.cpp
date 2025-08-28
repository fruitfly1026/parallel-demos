#include <iostream>
#include <omp.h>

using namespace std;

// int main() {
//     cout << "the begin of loop" << endl;

//     #pragma omp parallel for
//     for (int i = 0; i < 10; ++i) {
//         cout << i;
//     }
//     cout << endl << "the end of loop" << endl;

// //    Expected output will be similar to this:
// //    the begin of loop
// //    6378049152
// //    the end of loop
//     return 0;
// }

int main() {
    cout << "the begin of loop" << endl;
    
    #pragma omp parallel for
    for (int i = 0; i < 10; ++i) {
        // cout << i << endl;
        int tid = omp_get_thread_num();
        // cout << tid;
        #pragma omp critical 
        {
            cout << "(" << tid << "," << i << ") ";
        }
    }
    cout << endl << "the end of loop" << endl;

    return 0;
}