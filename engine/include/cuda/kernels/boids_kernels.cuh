#ifndef __BOIDS_KERNELS_CUH__
#define __BOIDS_KERNELS_CUH__

#include "vector_types.h"

namespace BoidsKernels
{
	__device__ int3 calcGridPosition(float4 pos, int3 grid, float3 gridSize);

	__device__ int calcCellIndex(int3 gridPos, int3 grid);

	__device__ int calcGridHash(int3 gridPos, int numBuckets);

	__global__ void build_spatial_grid
		(
			float4 *pos, 
			int *boidsIndex, 
			int *cellIndex, 
			int numBoids, 
			int3 grid,
			float3 gridSize
		);

	__global__ void reorder_boids
		(
			float4 *pos,
			float4 *spos,
			float4 *vel,
			float4 *svel,
			int *cellStartIndex,
			int *cellEndIndex,
			int *cellIndex,
			int *boidsIndex,
			int numBoids
		);

	__global__ void calculate_boids_direction
		(
			float4 *pos,  
			float4 *vel,
			float4 *scratch,
			int *cellStartIndex,
			int *cellEndIndex,
			int *cellIndex,
			int *boidsIndex,
			int numBoids,
			int3 grid
		);

	__global__ void update_boids
		(
			float4 *pos, 
			float4 *vel, 
			float4 *scratch,
			float *model,
			float dt, 
			float h,
			int numBoids,
			float3 gridSize
		);

}


#endif