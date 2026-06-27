NVCC      = nvcc
NVCCFLAGS = -O2 -arch=sm_70    # adjust sm_XX to your GPU (V100=70, A100=80, H100=90)
BINS      = 01_hello 02_vector_add 03_saxpy 04_matmul

all: $(BINS)

%: %.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

clean:
	rm -f $(BINS)
