#include <iostream>
#include <random>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/unique.h>

#include "../../include/cuda/BoidsDeviceData.cuh"
#include "../../include/cuda/cuda_util.h"
#include "../../include/cuda/kernels/boids_kernels.cuh"

using namespace PhysicsEngine;
using namespace BoidsKernels;

void PhysicsEngine::allocateBoidsDeviceData(BoidsDeviceData* boids)
{
	int numBoids = boids->numBoids;
	int numCells = boids->numCells;

	// allocate memory on host
	boids->h_pos = new float4[numBoids];
	boids->h_vel = new float4[numBoids];
	boids->h_spos = new float4[numBoids];
	boids->h_svel = new float4[numBoids];
	
	boids->h_cellStartIndex = new int[numCells];
	boids->h_cellEndIndex = new int[numCells];
	boids->h_cellIndex = new int[numBoids];

	// allocate memory on device
	gpuErrchk(cudaMalloc((void**)&(boids->d_pos), numBoids*sizeof(float4)));
	gpuErrchk(cudaMalloc((void**)&(boids->d_vel), numBoids*sizeof(float4)));
	gpuErrchk(cudaMalloc((void**)&(boids->d_spos), numBoids*sizeof(float4)));
	gpuErrchk(cudaMalloc((void**)&(boids->d_svel), numBoids*sizeof(float4)));

	gpuErrchk(cudaMalloc((void**)&(boids->d_cellStartIndex), numCells*sizeof(int)));
	gpuErrchk(cudaMalloc((void**)&(boids->d_cellEndIndex), numCells*sizeof(int)));
	gpuErrchk(cudaMalloc((void**)&(boids->d_cellHash), numBoids*sizeof(int)));
}

void PhysicsEngine::deallocateBoidsDeviceData(BoidsDeviceData* boids)
{
	// free memory on host
	delete [] boids->h_pos;
	delete [] boids->h_vel;
	delete [] boids->h_spos;
	delete [] boids->h_svel;

	delete [] boids->h_cellStartIndex;
	delete [] boids->h_cellEndIndex;
	delete [] boids->h_cellIndex;

	// free memory on device
	gpuErrchk(cudaFree(boids->d_pos));
	gpuErrchk(cudaFree(boids->d_vel));
	gpuErrchk(cudaFree(boids->d_spos));
	gpuErrchk(cudaFree(boids->d_svel));

	gpuErrchk(cudaFree(boids->d_cellStartIndex));
	gpuErrchk(cudaFree(boids->d_cellEndIndex));
	gpuErrchk(cudaFree(boids->d_cellHash));
}

void PhysicsEngine::initializeBoidsDeviceData(BoidsDeviceData* boids)
{

}

void PhysicsEngine::updateBoidsDeviceData(BoidsDeviceData* boids)
{
	dim3 gridSize(256,1,1);
	dim3 blockSize(256,1,1);
	//set_array_to_value<int> <<< gridSize, blockSize >>>(boids->d_cellStartIndex, -1, boids->numCells);
	//set_array_to_value<int> <<< gridSize, blockSize >>>(boids->d_cellEndIndex, -1, boids->numCells);

	// build_spatial_grid <<< gridSize, blockSize >>>
	// (
	// 	boids->d_pos, 
	// 	boids->d_particleIndex, 
	// 	boids->d_cellHash, 
	// 	boids->numParticles, 
	// 	boids->particleGridDim,
	// 	boids->particleGridSize
	// );

	// thrust::device_ptr<int> t_a(boids->d_cellHash);
	// thrust::device_ptr<int> t_b(boids->d_particleIndex);
	// thrust::sort_by_key(t_a, t_a + boids->numParticles, t_b);

	// reorder_particles <<< gridSize, blockSize >>>
	// (
	// 	boids->d_pos,
	// 	boids->d_spos,
	// 	boids->d_vel,
	// 	boids->d_svel,
	// 	boids->d_particleType,
	// 	boids->d_sparticleType,
	// 	boids->d_cellStartIndex,
	// 	boids->d_cellEndIndex,
	// 	boids->d_cellHash,
	// 	boids->d_particleIndex,
	// 	boids->numParticles
	// );

	// gpuErrchk(cudaMemcpy(&((boids->particles)[0]), boids->d_output, 3*boids->numParticles*sizeof(float), cudaMemcpyDeviceToHost));
	// gpuErrchk(cudaMemcpy(boids->h_pos, boids->d_pos, boids->numParticles*sizeof(float4), cudaMemcpyDeviceToHost));
	// gpuErrchk(cudaMemcpy(boids->h_rho, boids->d_rho, boids->numParticles*sizeof(float), cudaMemcpyDeviceToHost));
}