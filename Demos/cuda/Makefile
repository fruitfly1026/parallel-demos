NVCC      = nvcc
NVCCFLAGS = -O3 -arch=sm_86   # adjust sm_XX to your GPU (RTX A4500/Ampere=86, V100=70, A100=80, H100=90)
BINS      = 01_hello 02_vector_add 03_saxpy 04_matmul

all: $(BINS)

%: %.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

clean:
	rm -f $(BINS)
