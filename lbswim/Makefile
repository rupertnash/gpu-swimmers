CUDA_ARCH = -arch=sm_35 
CUDA_FLAGS = --compiler-options '-fPIC'
CUDA_LIBS = -lcudart

cuda_lib_sources = lbswim.cu
cuda_lib_objects = $(subst .cu,.o,$(cuda_lib_sources)) 

# pull in dependency info for *existing* .o files
-include $(cuda_lib_objects:.o=.d)

main : main.o liblbswim.so
	g++ $< -llbswim -o $@

liblbswim.so : $(cuda_lib_objects)
	g++ --shared $^ $(CUDA_LIBS) -o $@

%.o : %.cu
	nvcc -c $(CUDA_ARCH) $(CUDA_FLAGS) $*.cu
	nvcc -M $(CUDA_ARCH) $(CUDA_FLAGS) $*.cu | grep -v '\/usr' > $*.d

.PHONEY: clean
clean:
	-rm -rf main liblbswim.so *.o *.d
