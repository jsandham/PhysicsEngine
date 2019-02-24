#include "../../../include/cuda/kernels/boids_kernels.cuh"

#include "../../../include/cuda/helper_math.h"

#include <stdio.h>

using namespace BoidsKernels;

#define PI 3.14159265358979323846264f

__device__ const int blockSize = 256;
// __device__ const float f1 = 315.0f/(64.0f*PI);
// __device__ const float f2 = 45.0f/PI;
// __device__ const float nu = 0.05f;// 3.5 * 0.0075f;


__device__ int3 BoidsKernels::calcGridPosition(float4 pos, int3 grid, float3 gridSize)
{
	int3 gridPosition;
	/*gridPosition.x = __float2int_rd(pos.x * grid.x / gridSize.x);
	gridPosition.y = __float2int_rd(pos.y * grid.y / gridSize.y);
	gridPosition.z = __float2int_rd(pos.z * grid.z / gridSize.z);*/
	gridPosition.x = floor(pos.x * grid.x / gridSize.x);
	gridPosition.y = floor(pos.y * grid.y / gridSize.y);
	gridPosition.z = floor(pos.z * grid.z / gridSize.z);

	return gridPosition;
}


// what index the given gridPos corresponds to in the 1D array of cells
__device__ int BoidsKernels::calcCellIndex(int3 gridPos, int3 grid)
{
	return grid.y * grid.x * gridPos.z + grid.x * gridPos.y + gridPos.x;
}


// calculate spatial has for infinite domains
__device__ int BoidsKernels::calcGridHash(int3 gridPos, int numBuckets)
{
	const uint p1 = 73856093;
	const uint p2 = 19349663;
	const uint p3 = 83492791;
	int n = p1*gridPos.x ^ p2*gridPos.y ^ p3*gridPos.z;
	n %= numBuckets;
	return n;
}


__global__ void BoidsKernels::build_spatial_grid
(
	float4 *pos,
	int *particleIndex,
	int *cellIndex,
	int numParticles,
	int3 grid,
	float3 gridSize
)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int offset = 0;

	while (index + offset < numParticles){
		// find which grid cell the particle is in
		int3 gridPos = calcGridPosition(pos[index + offset], grid, gridSize);

		// compute cell index 
		int cindex = calcCellIndex(gridPos, grid);

		particleIndex[index + offset] = index + offset;
		cellIndex[index + offset] = cindex;

		offset += blockDim.x*gridDim.x;
	}
}

__global__ void BoidsKernels::reorder_particles
(
	float4 *pos,
	float4 *spos,
	float4 *vel,
	float4 *svel,
	int *particleType,
	int *sparticleType,
	int *cellStartIndex,
	int *cellEndIndex,
	int *cellIndex,
	int *particleIndex,
	int numParticles
)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int offset = 0;

	__shared__ int sharedcellIndex[blockSize + 1];  //blockSize + 1

	while (index + offset < numParticles){
		sharedcellIndex[threadIdx.x] = cellIndex[index + offset];
		if (threadIdx.x == blockDim.x - 1)
		{
			if (index + offset + 1 < numParticles){
				sharedcellIndex[threadIdx.x + 1] = cellIndex[index + offset + 1];
			}
			else{
				sharedcellIndex[threadIdx.x + 1] = -1;
			}
		}

		__syncthreads();

		if (sharedcellIndex[threadIdx.x] != sharedcellIndex[threadIdx.x + 1]){
			cellStartIndex[sharedcellIndex[threadIdx.x + 1]] = index + offset + 1;
			// cellEndIndex[sharedcellIndex[threadIdx.x]] = index + offset;
			cellEndIndex[sharedcellIndex[threadIdx.x]] = index + offset + 1;
		}

		// reorder position and velocity
		int p = particleIndex[index + offset];
		spos[index + offset] = pos[p];
		svel[index + offset] = vel[p];
		sparticleType[index + offset] = particleType[p];

		offset += blockDim.x*gridDim.x;
	}

	__syncthreads();

	if (threadIdx.x == 0 && blockIdx.x == 0){
		cellStartIndex[sharedcellIndex[0]] = 0;
	}
}