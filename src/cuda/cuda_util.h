#ifndef __CUDA_UTIL_H__
#define __CUDA_UTIL_H__

#include <stdio.h>

// ==========================================================================================
// CUDA ERROR CHECKING CODE
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) getchar();
   }
}

// ==========================================================================================

#endif
