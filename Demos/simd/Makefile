# SIMD demos - Intel AVX2 / AVX-512 (CSC 548, Topic 5)
# Verified on Intel Xeon w7-2495X, gcc 13.3, -march=native (enables AVX-512).
CC       = gcc
# -march=native turns on AVX-512 on capable CPUs; -mavx2 -mfma keep an explicit
# AVX2+FMA floor so the intrinsic paths build even without native tuning.
CFLAGS   = -O3 -march=native -mavx2 -mfma
LDLIBS   = -lm
BINS     = 01_vector_add 02_saxpy 03_dot_product 04_auto_vectorize

all: $(BINS) 04_novec

%: %.c
	$(CC) $(CFLAGS) -o $@ $< $(LDLIBS)

# Contrast build for demo 04: compiler auto-vectorization forced OFF.
04_novec: 04_auto_vectorize.c
	$(CC) -O3 -fno-tree-vectorize -o $@ $< $(LDLIBS)

# AVX2-only baseline builds (portable floor, no AVX-512):
avx2: CFLAGS = -O3 -mavx2 -mfma
avx2: $(BINS)

# Show which loops the compiler vectorized in demo 04:
vecreport: 04_auto_vectorize.c
	$(CC) $(CFLAGS) -fopt-info-vec -o /dev/null $< $(LDLIBS)

run: all
	@for b in $(BINS); do echo "===== $$b ====="; ./$$b; echo; done

clean:
	rm -f $(BINS) 04_novec

.PHONY: all avx2 vecreport run clean
