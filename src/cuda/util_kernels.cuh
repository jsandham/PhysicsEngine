#ifndef __UTILKERNELS_CUH__
#define __UTILKERNELS_CUH__

#include "vector_types.h"


template<typename T>
__global__ void set_variable_to_value(T* variable, T value)
{
	if(threadIdx.x + blockIdx.x*blockDim.x){
		*variable = value;
	}
}


template<typename T>
__global__ void set_array_to_value(T array[], T value, int n)
{
	int index = threadIdx.x + blockIdx.x*blockDim.x;
	int offset = 0;

	while(index + offset < n){
		array[index + offset] = value;

		offset += blockDim.x*gridDim.x;
	}
}


template<typename T>
__global__ void copy_array_to_array(T input[], T output[], int n)
{
	int index = threadIdx.x + blockIdx.x*blockDim.x;
	int offset = 0;

	while(index + offset < n){
		output[index + offset] = input[index + offset];

		offset += blockDim.x*gridDim.x;
	}
}

template<typename T>
__global__ void swap_arrays(T array1[], T array2[], int n)
{
	int index = threadIdx.x + blockIdx.x*blockDim.x;
	int offset = 0;

	while (index + offset < n){
		T temp1 = array1[index + offset];

		array1[index + offset] = array2[index + offset];
		array2[index + offset] = temp1;

		offset += blockDim.x*gridDim.x;
	}
}




#endif
