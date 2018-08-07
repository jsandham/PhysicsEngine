#ifndef __FLUID_KERNELS_CUH__
#define __FLUID_KERNELS_CUH__

#include "vector_types.h"

__device__ int3 calcGridPosition(float4 pos, int3 grid, float3 gridSize);

__device__ int calcCellIndex(int3 gridPos, int3 grid);

__device__ int calcGridHash(int3 gridPos, int numBuckets);

__device__ void boundary(float4 *pos, float4 *vel, float h);

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

__global__ void calculate_fluid_particle_density
	(
		float4 *pos, 
		float *rho, 
		int *particleType,
		int *cellStartIndex,
		int *cellEndIndex,
		int *cellIndex,
		int *particleIndex,
		int numParticles,
		float h2,
		float h9,
		int3 grid
	);

__global__ void calculate_solid_particle_density
	(
		float4 *pos,
		float *rho,
		int *particleType,
		int *cellStartIndex,
		int *cellEndIndex,
		int *cellIndex,
		int *particleIndex,
		int numParticles,
		float h2,
		float h9,
		int3 grid
	);

__global__ void calculate_pressure
	(
		float *rho,
		float *rho0,
		float *pres,
		int numParticles,
		float kappa
	);

__global__ void apply_pressure_and_gravity_acceleration
	(
		float4 *pos, 
		float4 *vel,
		float *rho,
		float *pres,
		int *particleType,
		int *cellStartIndex,
		int *cellEndIndex,
		int *cellIndex,
		int *particleIndex,
		int numParticles,
		float dt,
		float h,
		float h6,
		int3 grid
	);

__global__ void compute_solid_particle_velocity
	(
		float4 *pos,
		float4 *vel,
		int *particleType,
		int numParticles
	);

__global__ void apply_xsph_viscosity
	(
		float4 *pos,
		float4 *vel,
		float *rho,
		int *particleType,
		int *cellStartIndex,
		int *cellEndIndex,
		int *cellIndex,
		int *particleIndex,
		int numParticles,
		float dt,
		float h,
		float h6,
		int3 grid
	);

__global__ void update_particles
	(
		float4 *pos, 
		float4 *vel, 
		int *particleType,
		float dt, 
		float h,
		int numParticles,
		float3 gridSize
	);


__global__ void copy_sph_arrays
	(
		float4 *pos,
		float4 *spos,
		float4 *vel,
		float4 *svel,
		int *particleType,
		int *sparticleType,
		float *output,
		int numParticles
	);
















__global__ void reorder_particles2
	(
		float4 *pos,
		float4 *oldPos,
		float4 *spos,
		float4 *soldPos,
		float4 *vel,
		float4 *oldVel,
		float4 *svel,
		float4 *soldVel,
		int *particleType,
		int *sparticleType,
		int *cellStartIndex,
		int *cellEndIndex,
		int *cellIndex,
		int *particleIndex,
		int numParticles
	);

__device__ float3 collideParticles(float4 pos1, float4 pos2, float4 vel1, float4 vel2, float radius1, float radius2, float spring, float damping, float shear, float attraction);

__global__ void	calculate_collisions
	(
		float4 *pos,
		float4 *oldPos,
		float4 *vel,
		float4 *oldVel,
		int *cellStartIndex,
		int *cellEndIndex,
		int *cellIndex,
		int *particleIndex,
		int numParticles,
		int3 grid
	);

__global__ void update_particles2
	(
		float4 *pos,
		float4 *vel,
		float4 *oldPos,
		float4 *oldVel,
		float dt,
		int numParticles,
		float3 gridSize
	);

__global__ void copy_arrays
	(
		float4 *pos,
		float4 *oldPos,
		float4 *vel,
		float4 *oldVel,
		int *particleType,
		int *sparticleType,
		float *output,
		int numParticles
	);

//__global__ void copy_arrays
//	(
//		float4 *pos,
//		float4 *oldPos,
//		float4 *spos,
//		float4 *soldPos,
//		float4 *vel,
//		float4 *oldVel,
//		float4 *svel,
//		float4 *soldVel,
//		int *particleType,
//		int *sparticleType,
//		float *output,
//		int numParticles
//	);

#endif