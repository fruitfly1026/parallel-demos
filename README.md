# parallel-demos

Small, self-contained code demos for **CSC 548 — Parallel Systems** (NCSU), meant to be
run live in class. Each folder has its own `README.md` with build/run commands.

## Code demos (under `Demos/`)
| 548 Topic | Folder | What's inside |
|-----------|--------|---------------|
| 3 · Message Passing | [`Demos/mpi`](Demos/mpi) | hello, send/recv, ping-pong latency, collectives, parallel pi, deadlock + fixes |
| 4 · Shared Memory | [`Demos/openmp-examples`](Demos/openmp-examples) | OpenMP examples |
| 5 · Vector (SIMD) | [`Demos/simd`](Demos/simd) | AVX intrinsics: vector add, SAXPY (FMA), dot product, auto-vectorization |
| 6 · GPU (CUDA) | [`Demos/cuda`](Demos/cuda) | hello, vector add, SAXPY (grid-stride), naive vs tiled matmul |
| 10 · MapReduce & Spark | [`Demos/spark`](Demos/spark) | word count, Monte-Carlo pi, transformations & lineage |

Most demos target the **ARC** cluster (x86 CPUs + GPUs). See each folder's README for the
`module load`, `make`, and `mpirun`/`srun`/`spark-submit` lines.

```bash
# typical flow on ARC
cd Demos/mpi   && module load openmpi && make && mpirun -n 4 ./01_hello
cd Demos/simd  && module load gcc     && make && ./01_vector_add
cd Demos/cuda  && module load cuda    && make && srun --gres=gpu:1 ./02_vector_add
cd Demos/spark && module load spark   && spark-submit --master 'local[*]' 01_wordcount.py
```
