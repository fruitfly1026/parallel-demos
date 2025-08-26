### Note that: this script provides the commands you need. You might not be able to directly run this script to complete all tests. ###

# Run sequential code
gcc -O2 -std=c11 main.c seq.c -o seq -lm

# Run OpenMP code
export OMP_NUM_THREADS=4  # TODO: You can change the number of threads as needed
gcc -O2 -fopenmp -std=c11 main.c omp.c -o omp -lm

# Execute the programs
echo "Running sequential version..."
./seq 10000000  # You can change the number of samples as needed 

echo "Running OpenMP version..."
./omp 10000000  # You can change the number of samples as needed
