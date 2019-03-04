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
#include "../../include/cuda/kernels/util_kernels.cuh"

using namespace PhysicsEngine;
using namespace BoidsKernels;

void PhysicsEngine::allocateBoidsDeviceData(BoidsDeviceData* boids)
{
	int numBoids = boids->numBoids;
	int numVoxels = boids->numVoxels;

	// allocate memory on host
	boids->h_pos = new float4[numBoids];
	boids->h_vel = new float4[numBoids];
	boids->h_spos = new float4[numBoids];
	boids->h_svel = new float4[numBoids];

	boids->h_cellHash = new int[numBoids];
	boids->h_boidsIndex = new int[numBoids], 
	
	boids->h_cellStartIndex = new int[numVoxels];
	boids->h_cellEndIndex = new int[numVoxels];

	boids->h_modelMatrices = new float[numBoids*16];

	// allocate memory on device
	gpuErrchk(cudaMalloc((void**)&(boids->d_pos), numBoids*sizeof(float4)));
	gpuErrchk(cudaMalloc((void**)&(boids->d_vel), numBoids*sizeof(float4)));
	gpuErrchk(cudaMalloc((void**)&(boids->d_spos), numBoids*sizeof(float4)));
	gpuErrchk(cudaMalloc((void**)&(boids->d_svel), numBoids*sizeof(float4)));
	gpuErrchk(cudaMalloc((void**)&(boids->d_scratch), numBoids*sizeof(float4)));

	gpuErrchk(cudaMalloc((void**)&(boids->d_cellHash), numBoids*sizeof(int)));
	gpuErrchk(cudaMalloc((void**)&(boids->d_boidsIndex), numBoids*sizeof(int)));

	gpuErrchk(cudaMalloc((void**)&(boids->d_cellStartIndex), numVoxels*sizeof(int)));
	gpuErrchk(cudaMalloc((void**)&(boids->d_cellEndIndex), numVoxels*sizeof(int)));

	gpuErrchk(cudaMalloc((void**)&(boids->d_modelMatrices), 16*numBoids*sizeof(float)));
}

void PhysicsEngine::deallocateBoidsDeviceData(BoidsDeviceData* boids)
{
	// free memory on host
	delete [] boids->h_pos;
	delete [] boids->h_vel;
	delete [] boids->h_spos;
	delete [] boids->h_svel;

	delete [] boids->h_cellHash;
	delete [] boids->h_boidsIndex;

	delete [] boids->h_cellStartIndex;
	delete [] boids->h_cellEndIndex;

	delete [] boids->h_modelMatrices;

	// free memory on device
	gpuErrchk(cudaFree(boids->d_pos));
	gpuErrchk(cudaFree(boids->d_vel));
	gpuErrchk(cudaFree(boids->d_spos));
	gpuErrchk(cudaFree(boids->d_svel));
	gpuErrchk(cudaFree(boids->d_scratch));

	gpuErrchk(cudaFree(boids->d_cellHash));
	gpuErrchk(cudaFree(boids->d_boidsIndex));

	gpuErrchk(cudaFree(boids->d_cellStartIndex));
	gpuErrchk(cudaFree(boids->d_cellEndIndex));

	gpuErrchk(cudaFree(boids->d_modelMatrices));
}

void PhysicsEngine::initializeBoidsDeviceData(BoidsDeviceData* boids)
{
	int numBoids = boids->numBoids;
	int numVoxels = boids->numVoxels;

	std::cout << "number of boids: " << boids->numBoids << " number of voxels: " << boids->numVoxels << std::endl;

	std::random_device rd;
    std::mt19937 generator(rd());
    std::uniform_real_distribution<> distribution(0, 1);

	for(unsigned int i = 0; i < numBoids; i++){
    	float x = boids->voxelGridSize.x * (float)distribution(generator);
    	float y = boids->voxelGridSize.y * (float)distribution(generator);
    	float z = boids->voxelGridSize.z * (float)distribution(generator);

    	std::cout << "x: " << x << " y: " << y << " z: " << z << std::endl;

		boids->h_pos[i] = make_float4(x, y, z, 0.0f);
		boids->h_vel[i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
		boids->h_spos[i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
		boids->h_svel[i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

		boids->h_cellHash[i] = 0;
		boids->h_boidsIndex[i] = 0;
	}

	for(unsigned int i = 0; i < numVoxels; i++){
		boids->h_cellStartIndex[i] = -1;
		boids->h_cellEndIndex[i] = -1;
	}

	// copy to device
	gpuErrchk(cudaMemcpy(boids->d_pos, boids->h_pos, numBoids * sizeof(float4), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(boids->d_vel, boids->h_vel, numBoids * sizeof(float4), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(boids->d_spos, boids->h_spos, numBoids * sizeof(float4), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(boids->d_svel, boids->h_svel, numBoids * sizeof(float4), cudaMemcpyHostToDevice));

	gpuErrchk(cudaMemcpy(boids->d_cellHash, boids->h_cellHash, numBoids * sizeof(int), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(boids->d_boidsIndex, boids->h_boidsIndex, numBoids * sizeof(int), cudaMemcpyHostToDevice));

	gpuErrchk(cudaMemcpy(boids->d_cellStartIndex, boids->h_cellStartIndex, numVoxels * sizeof(int), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(boids->d_cellEndIndex, boids->h_cellEndIndex, numVoxels * sizeof(int), cudaMemcpyHostToDevice));
}

void PhysicsEngine::updateBoidsDeviceData(BoidsDeviceData* boids)
{
	dim3 gridSize(16,1,1);
	dim3 blockSize(16,1,1);
	set_array_to_value<int> <<< gridSize, blockSize >>>(boids->d_cellStartIndex, -1, boids->numVoxels);
	set_array_to_value<int> <<< gridSize, blockSize >>>(boids->d_cellEndIndex, -1, boids->numVoxels);

	build_spatial_grid <<< gridSize, blockSize >>>
	(
		boids->d_pos, 
		boids->d_boidsIndex, 
		boids->d_cellHash, 
		boids->numBoids, 
		boids->voxelGridDim,
		boids->voxelGridSize
	);

	gpuErrchk(cudaMemcpy(boids->h_boidsIndex, boids->d_boidsIndex, boids->numBoids*sizeof(int), cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(boids->h_cellHash, boids->d_cellHash, boids->numBoids*sizeof(int), cudaMemcpyDeviceToHost));

	// for(int i = 0; i < boids->numBoids; i++){
	// 	std::cout << boids->h_boidsIndex[i] << "  " << boids->h_cellHash[i] << std::endl;
	// }	

	thrust::device_ptr<int> t_a(boids->d_cellHash);
	thrust::device_ptr<int> t_b(boids->d_boidsIndex);
	thrust::sort_by_key(t_a, t_a + boids->numBoids, t_b);

	reorder_boids <<< gridSize, blockSize >>>
	(
		boids->d_pos,
		boids->d_spos,
		boids->d_vel,
		boids->d_svel,
		boids->d_cellStartIndex,
		boids->d_cellEndIndex,
		boids->d_cellHash,
		boids->d_boidsIndex,
		boids->numBoids
	);

	calculate_boids_direction <<< gridSize, blockSize >>>
	(
		boids->d_spos, 
		boids->d_svel, 
		boids->d_scratch, 
		boids->d_cellStartIndex,
		boids->d_cellEndIndex,
		boids->d_cellHash,
		boids->d_boidsIndex,
		boids->numBoids,
		boids->voxelGridDim
	);

	update_boids<<< gridSize, blockSize >>>
	(
		boids->d_spos,
		boids->d_svel,
		boids->d_scratch,
		boids->d_modelMatrices,
		0.1f,
		boids->h,
		boids->numBoids,
		boids->voxelGridSize
	);

	gpuErrchk(cudaMemcpy(boids->h_modelMatrices, boids->d_modelMatrices, 16*boids->numBoids*sizeof(float), cudaMemcpyDeviceToHost));
}