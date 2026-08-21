# Built and verified with OpenMPI 4.1.6 (mpicc) on a 48-core Intel Xeon w7-2495X.
CC      = mpicc
CFLAGS  = -O3 -Wall
LDLIBS  = -lm
BINS    = 01_hello 02_send_recv 03_ping_pong 04_collectives 05_pi 06_deadlock

all: $(BINS)

%: %.c
	$(CC) $(CFLAGS) -o $@ $< $(LDLIBS)

# Quick smoke test of every demo (see each .c for the embedded reference run).
#   mpirun --oversubscribe -np 4 ./01_hello
#   mpirun --oversubscribe -np 2 ./02_send_recv
#   mpirun --oversubscribe -np 2 ./03_ping_pong
#   mpirun --oversubscribe -np 8 ./04_collectives
#   mpirun --oversubscribe -np 8 ./05_pi
#   timeout 8 mpirun --oversubscribe -np 4 ./06_deadlock deadlock   # hangs (deadlock demo)
#   mpirun --oversubscribe -np 4 ./06_deadlock fix                  # completes

clean:
	rm -f $(BINS)
