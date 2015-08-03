c_sources = 
cuda_sources = lb.cu main.cu SwimmerArray.cu

objects = $(subst .cpp,.o,$(c_sources)) $(subst .cu,.o,$(cuda_sources)) 

CC = nvcc

%.o : %.cu
	$(CC) -c $<

swimmers : $(objects)
	$(CC) $^ -o $@
