#ifndef __CLOTH_KERNELS_CUH__
#define __CLOTH_KERNELS_CUH__

#include "vector_types.h"

__global__ void calculate_forces
	(
		float4 *pos,
		float4 *oldPos,
		float4 *acc,
		float mass,
		float kappa,
		float c,
		float dt,
		int nx,
		int ny
	);

__global__ void verlet_integration
	(
		float4 *pos,
		float4 *oldPos,
		float4 *acc,
		float *output,
		float dt,
		int nx,
		int ny
	);

__global__ void update_triangle_mesh
	(
		float4 *pos,
		int *triangleIndices,
		float *triangleVertices,
		int nx,
		int ny
	);






__global__ void apply_constraints
	(
		float4 *pos,
		int nx,
		int ny
	);

__global__ void calculate_forces2
	(
		float4 *pos,
		float4 *oldPos,
		float4 *acc,
		float mass,
		float kappa,
		float c,
		float dt,
		int nx,
		int ny
	);

#endif