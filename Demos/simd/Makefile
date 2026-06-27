CC      = gcc
CFLAGS  = -O3 -march=native -fopenmp
BINS    = 01_vector_add 02_saxpy 03_dot_product 04_auto_vectorize

all: $(BINS)

%: %.c
	$(CC) $(CFLAGS) -o $@ $<

clean:
	rm -f $(BINS)
