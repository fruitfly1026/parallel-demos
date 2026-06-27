# MPI demos

Small, self-contained MPI programs for in-class demos (CSC 548, Topic 3 — Message Passing).

| File | Shows |
|------|-------|
| `01_hello.c` | `MPI_Init/Comm_rank/Comm_size`, processor name |
| `02_send_recv.c` | blocking point-to-point `MPI_Send`/`MPI_Recv` |
| `03_ping_pong.c` | round-trip latency vs message size (cf. HW1) |
| `04_collectives.c` | `MPI_Bcast`, `MPI_Reduce`, `MPI_Allreduce` |
| `05_pi.c` | parallel integration with one `MPI_Reduce` |
| `06_deadlock.c` | a classic deadlock + two fixes (`Sendrecv`, nonblocking) |

## Build & run (ARC)
```bash
module load openmpi        # or the MPI module on your system
make
mpirun -n 4 ./01_hello
mpirun -n 2 ./03_ping_pong
```
On a Slurm batch/interactive node use `srun -n 4 ./01_hello`.
