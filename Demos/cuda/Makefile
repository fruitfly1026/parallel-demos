# CUDA demos, organised one folder per inefficiency pattern.
# Each folder holds a matched pair: the unoptimized version and the fix.
#
# -lineinfo is required for cuThermo (and Nsight) to map sectors back to
# source lines; keep it on.
NVCC      = nvcc
NVCCFLAGS = -O3 -arch=sm_86 -lineinfo
# adjust -arch for your GPU: RTX A4500/Ampere=86, RTX 4090=89,
#                            V100=70, A100=80, H100=90

SRCS := $(wildcard 01_hello.cu */*.cu)
BINS := $(SRCS:.cu=)

all: $(BINS)

%: %.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

clean:
	rm -f $(BINS)

.PHONY: all clean
