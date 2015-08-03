cuda_sources = lb.cu SwimmerArray.cu

objects = $(subst .cu,.o,$(cuda_sources)) 

CFLAGS = -arch=sm_35 --shared --compiler-options '-fPIC'

%.o : %.cu
	nvcc -dc $(CFLAGS) $<

liblbswim.so : $(objects)
	gcc -shared $^ -o $@
