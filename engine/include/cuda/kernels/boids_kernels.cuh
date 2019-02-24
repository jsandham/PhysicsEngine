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
			int *particlesIndex, 
			int *cellIndex, 
			int numParticles, 
			int3 grid,
			float3 gridSize
		);

	__global__ void reorder_particles
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
		);
}


#endif