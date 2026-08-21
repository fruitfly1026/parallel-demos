# MPI demos

Message-passing programs (CSC 548, Topic 3 — Message Passing). All six were
compiled with `mpicc -O3` (OpenMPI 4.1.6) and run on a 48-core **Intel Xeon
w7-2495X**; each `.c` file ends with an embedded reference-run comment showing
the actual output.

Every demo includes:
- a **correctness check** — the received / collective result vs a serial
  reference, printing PASS/FAIL;
- **parameter-varying** runs (`-np` 2/4/8/16, message sizes);
- where relevant, an **inefficient-vs-optimized** version timed with
  `MPI_Wtime`, with the inefficiency + fix explained in-code.

| File | Shows | **Inefficiency → fix** (timed) | Measured |
|------|-------|--------------------------------|----------|
| `01_hello.c` | `MPI_Init/Comm_rank/Comm_size`; verifies the ranks form the set {0..P-1} | — | ranks verified, np 2/4/8 |
| `02_send_recv.c` | blocking `MPI_Send`/`MPI_Recv`; receiver checks every element | — | sizes 1→65536 verified |
| `03_ping_pong.c` | round-trip latency vs message size | **N separate one-int messages** — per-message latency dominates → **one aggregated message** that amortizes the fixed startup cost | **172×** |
| `04_collectives.c` | `Bcast`/`Reduce`/`Allreduce`/`Scatter`-`Gather` | **hand-rolled root loop** (root sends to every rank / sums in a loop) = **O(P)** → **`MPI_Bcast`/`MPI_Reduce`** tree = **O(log P)** | Bcast **2.7×**, Reduce **1.75×** @ np16 (gap grows with P) |
| `05_pi.c` | parallel integration of ∫₀¹4/(1+x²)dx | **`MPI_Reduce` called inside the loop** (a collective every iteration) → **one final reduce** of each rank's local partial sum | **82–222×** (np 2→8), err ~1e-13 |
| `06_deadlock.c` | a classic deadlock **and** its fix | **every rank `MPI_Send`s a large buffer before `MPI_Recv`** — the rendezvous protocol can't progress, so all ranks block → **deadlock** → **`MPI_Sendrecv`** (or nonblocking `Isend`/`Irecv`, or odd/even send-recv ordering) | deadlock **hangs** (killed by `timeout 8`); fix completes |

`06_deadlock` takes a mode argument:
```bash
timeout 8 mpirun --oversubscribe -np 4 ./06_deadlock deadlock   # hangs, then times out
mpirun --oversubscribe -np 4 ./06_deadlock fix                  # completes, verified
```

## Build & run
```bash
# this machine (OpenMPI 4.1.6 built under /mnt/data/jli/sw):
export PATH=/mnt/data/jli/sw/openmpi/bin:$PATH
export LD_LIBRARY_PATH=/mnt/data/jli/sw/openmpi/lib:$LD_LIBRARY_PATH
#   ...or `module load openmpi` on an ARC/Slurm cluster.
make
mpirun --oversubscribe -np 4 ./01_hello
mpirun --oversubscribe -np 4 ./05_pi
```
On a Slurm batch/interactive node use `srun -n 4 ./01_hello`.
