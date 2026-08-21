CC      = gcc
CFLAGS  = -O3 -fopenmp -Wall
LDLIBS  = -lm
BINS    = 01_hello 02_parallel_for 03_reduction 04_schedule 05_false_sharing

all: $(BINS)

%: %.c
	$(CC) $(CFLAGS) -o $@ $< $(LDLIBS)

clean:
	rm -f $(BINS)
