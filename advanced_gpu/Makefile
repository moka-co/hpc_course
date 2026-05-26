# Makefile for gpuadvanced CUDA programs
# Requires: CUDA toolkit (nvcc)

NVCC      := nvcc
NVCCFLAGS := -O2 -arch=sm_50

TARGETS := outer_product stencil dot_product row_scaling

.PHONY: all clean

all: $(TARGETS)

outer_product: outer_product.cu
	$(NVCC) $(NVCCFLAGS) -o $@.o $<

stencil: stencil.cu
	$(NVCC) $(NVCCFLAGS) -o $@.o $<

dot_product: dot_product.cu
	$(NVCC) $(NVCCFLAGS) -o $@.o $<

row_scaling: row_scaling.cu
	$(NVCC) $(NVCCFLAGS) -o $@.o $<

clean:
	rm -f $(TARGETS)
