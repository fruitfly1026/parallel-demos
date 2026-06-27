CC      = mpicc
CFLAGS  = -O2 -Wall
BINS    = 01_hello 02_send_recv 03_ping_pong 04_collectives 05_pi 06_deadlock

all: $(BINS)

%: %.c
	$(CC) $(CFLAGS) -o $@ $<

clean:
	rm -f $(BINS)
