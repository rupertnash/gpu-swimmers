// This is the combined CUDA source file to escape linking difficulties
#include "Shared.cu"
#include "lb.cu"
#include "SwimmerArray.cu"
template class SharedArray<double>;
template class SharedItem<CommonParams>;