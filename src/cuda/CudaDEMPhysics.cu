#include <iostream>
#include <random>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/unique.h>

#include "CudaDEMPhysics.cuh"
#include "kernels.cuh"
#include "util_kernels.cuh"
#include "cuda_util.cuh"

#include <stdio.h>

#include <cuda.h>

#include "Util.h"

CudaDEMPhysics::CudaDEMPhysics()
{
	grid = NULL;

	initCalled = false;
}

CudaDEMPhysics::~CudaDEMPhysics()
{
	// free old memory 
	if (initCalled){
		deallocateMemory();
	}
}

void CudaDEMPhysics::allocateMemory()
{
	// allocate memory on host
	h_pos = new float4[numParticles];
	h_vel = new float4[numParticles];
	h_oldPos = new float4[numParticles];
	h_oldVel = new float4[numParticles];
	//h_acc = new float4[numParticles];
	h_spos = new float4[numParticles];
	h_svel = new float4[numParticles];

	h_cellStartIndex = new int[numCells];
	h_cellEndIndex = new int[numCells];	
	h_cellIndex = new int[numParticles];
	h_particleIndex = new int[numParticles];
	h_particleType = new int[numParticles];
	h_sparticleType = new int[numParticles];

	// allocate memory on device
	gpuErrchk(cudaMalloc((void**)&d_pos, numParticles*sizeof(float4)));
	gpuErrchk(cudaMalloc((void**)&d_vel, numParticles*sizeof(float4)));
	gpuErrchk(cudaMalloc((void**)&d_oldPos, numParticles*sizeof(float4)));
	gpuErrchk(cudaMalloc((void**)&d_oldVel, numParticles*sizeof(float4)));
	//gpuErrchk(cudaMalloc((void**)&d_acc, numParticles*sizeof(float4)));
	gpuErrchk(cudaMalloc((void**)&d_spos, numParticles*sizeof(float4)));
	gpuErrchk(cudaMalloc((void**)&d_svel, numParticles*sizeof(float4)));
	gpuErrchk(cudaMalloc((void**)&d_soldPos, numParticles*sizeof(float4)));
	gpuErrchk(cudaMalloc((void**)&d_soldVel, numParticles*sizeof(float4)));

	gpuErrchk(cudaMalloc((void**)&d_output, 3 * numParticles*sizeof(float)));

	gpuErrchk(cudaMalloc((void**)&d_cellStartIndex, numCells*sizeof(int)));
	gpuErrchk(cudaMalloc((void**)&d_cellEndIndex, numCells*sizeof(int)));
	gpuErrchk(cudaMalloc((void**)&d_cellHash, numParticles*sizeof(int)));
	gpuErrchk(cudaMalloc((void**)&d_particleIndex, numParticles*sizeof(int)));
	gpuErrchk(cudaMalloc((void**)&d_particleType, numParticles*sizeof(int)));
	gpuErrchk(cudaMalloc((void**)&d_sparticleType, numParticles*sizeof(int)));
}

void CudaDEMPhysics::deallocateMemory()
{
	// free memory on host
	delete [] h_pos;
	delete [] h_vel;
	delete [] h_oldPos;
	delete [] h_oldVel;
	//delete [] h_acc;
	delete [] h_spos;
	delete [] h_svel;

	delete [] h_cellStartIndex;
	delete [] h_cellEndIndex;
	delete [] h_cellIndex;
	delete [] h_particleIndex;
	delete [] h_particleType;
	delete [] h_sparticleType;

	// free memory on device
	gpuErrchk(cudaFree(d_pos));
	gpuErrchk(cudaFree(d_vel));
	gpuErrchk(cudaFree(d_oldPos));
	gpuErrchk(cudaFree(d_oldVel));
	//gpuErrchk(cudaFree(d_acc));	
	gpuErrchk(cudaFree(d_spos));
	gpuErrchk(cudaFree(d_svel));
	gpuErrchk(cudaFree(d_soldPos));
	gpuErrchk(cudaFree(d_soldVel));

	gpuErrchk(cudaFree(d_output));

	gpuErrchk(cudaFree(d_cellStartIndex));
	gpuErrchk(cudaFree(d_cellEndIndex));
	gpuErrchk(cudaFree(d_cellHash));
	gpuErrchk(cudaFree(d_particleIndex));
	gpuErrchk(cudaFree(d_particleType));
	gpuErrchk(cudaFree(d_sparticleType));
}

void CudaDEMPhysics::init()
{
	if (particles.size() == 0){
		std::cout << "CudaFluidPhysicsEngine: Must set particles before calling init." << std::endl;
		return;
	}

	if (particleTypes.size() == 0){
		std::cout << "CudaFluidPhysicsEngine: Must set particle types before calling init." << std::endl;
		return;
	}

	if (grid == NULL){
		std::cout << "CudaFluidPhysicsEngine: Must set PhysicsEngine domain grid before calling init." << std::endl;
		return;
	}

	numParticles = particles.size() / 3;

	dt = 0.5f;

	numCells = grid->getNx() * grid->getNy() * grid->getNz();

	particleGridDim.x = grid->getNx();
	particleGridDim.y = grid->getNy();
	particleGridDim.z = grid->getNz();

	particleGridSize.x = grid->getX();
	particleGridSize.y = grid->getY();
	particleGridSize.z = grid->getZ();

	std::cout << "numCells: " << numCells << " particleGridDim.x: " << particleGridDim.x << " particleGridDim.y: " << particleGridDim.y << " particleGridDim.z: " << particleGridDim.z << std::endl;

	allocateMemory();

	for (int i = 0; i < numParticles; i++){
		h_pos[i].x = 0.0f;// particles[3 * i];
		h_pos[i].y = 0.0f;// particles[3 * i + 1];
		h_pos[i].z = 0.0f;// particles[3 * i + 2];
		h_pos[i].w = 0.0f;

		h_vel[i].x = 0.0f;
		h_vel[i].y = 0.0f;
		h_vel[i].z = 0.0f;
		h_vel[i].w = 0.0f;

		h_oldPos[i].x = particles[3 * i];
		h_oldPos[i].y = particles[3 * i + 1];
		h_oldPos[i].z = particles[3 * i + 2];
		h_oldPos[i].w = 0.0f;

		h_oldVel[i].x = 0.0f;
		h_oldVel[i].y = 0.0f;
		h_oldVel[i].z = 0.0f;
		h_oldVel[i].w = 0.0f;

		h_particleType[i] = particleTypes[i];
	}

	gpuErrchk(cudaMemcpy(d_pos, h_pos, numParticles*sizeof(float4), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_vel, h_vel, numParticles*sizeof(float4), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_oldPos, h_oldPos, numParticles*sizeof(float4), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_oldVel, h_oldVel, numParticles*sizeof(float4), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_particleType, h_particleType, numParticles*sizeof(int), cudaMemcpyHostToDevice));

	initCalled = true;
}

void CudaDEMPhysics::update()
{
	elapsedTime = 0;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start,0);

	uniformFiniteGridAlgorithm();

	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
}

void CudaDEMPhysics::setGridDomain(VoxelGrid *grid)
{
	this->grid = grid;
}

void CudaDEMPhysics::setParticles(std::vector<float> &particles)
{
	this->particles = particles;
}

void CudaDEMPhysics::setParticleTypes(std::vector<int> &particleTypes)
{
	this->particleTypes = particleTypes;
}

std::vector<float>& CudaDEMPhysics::getParticles()
{
	return particles;
}

std::vector<int>& CudaDEMPhysics::getParticleTypes()
{
	return particleTypes;
}


void CudaDEMPhysics::uniformFiniteGridAlgorithm()
{
	dim3 gridSize(64, 1, 1);
	dim3 blockSize(64, 1, 1);
	update_particles2 << < gridSize, blockSize >> >
		(
		d_pos,
		d_oldPos,
		d_vel,
		d_oldVel,
		dt,
		numParticles,
		particleGridSize
		);

	swap_arrays<float4> << < gridSize, blockSize >> >(d_oldPos, d_pos, numParticles);
	swap_arrays<float4> << < gridSize, blockSize >> >(d_oldVel, d_vel, numParticles);

	build_spatial_grid << < gridSize, blockSize >> >
		(
		d_oldPos,
		d_particleIndex,
		d_cellHash,
		numParticles,
		particleGridDim,
		particleGridSize
		);

	thrust::device_ptr<int> t_a(d_cellHash);
	thrust::device_ptr<int> t_b(d_particleIndex);
	thrust::sort_by_key(t_a, t_a + numParticles, t_b);

	reorder_particles << < gridSize, blockSize >> >
		(
		d_oldPos,
		d_pos,
		d_oldVel,
		d_vel,
		d_particleType,
		d_sparticleType,
		d_cellStartIndex,
		d_cellEndIndex,
		d_cellHash,
		d_particleIndex,
		numParticles
		);

	/*reorder_particles2 << < gridSize, blockSize >> >
		(
		d_pos,
		d_oldPos,
		d_spos,
		d_soldPos,
		d_vel,
		d_oldVel,
		d_svel,
		d_soldVel,
		d_particleType,
		d_sparticleType,
		d_cellStartIndex,
		d_cellEndIndex,
		d_cellHash,
		d_particleIndex,
		numParticles
		);*/

	for (int i = 0; i < 1; i++){
		calculate_collisions << < gridSize, blockSize >> >
			(
			d_oldPos,
			d_pos,
			d_oldVel,
			d_vel,
			d_cellStartIndex,
			d_cellEndIndex,
			d_cellHash,
			d_particleIndex,
			numParticles,
			particleGridDim
			);
		/*calculate_collisions << < gridSize, blockSize >> >
			(
			d_spos,
			d_soldPos,
			d_svel,
			d_soldVel,
			d_cellStartIndex,
			d_cellEndIndex,
			d_cellHash,
			d_particleIndex,
			numParticles,
			particleGridDim
			);*/

		//swap_arrays<float4> << < gridSize, blockSize >> >(d_soldVel, d_svel, numParticles);
		swap_arrays<float4> << < gridSize, blockSize >> >(d_oldVel, d_vel, numParticles);
	}

	copy_arrays << < gridSize, blockSize >> >
	(
		d_pos,
		d_oldPos,
		d_vel,
		d_oldVel,
		d_particleType,
		d_sparticleType,
		d_output,
		numParticles
	);

	/*copy_arrays << < gridSize, blockSize >> >
		(
		d_pos,
		d_oldPos,
		d_spos,
		d_soldPos,
		d_vel,
		d_oldVel,
		d_svel,
		d_soldVel,
		d_particleType,
		d_sparticleType,
		d_output,
		numParticles
		);*/

	set_array_to_value<int> << < gridSize, blockSize >> >(d_cellStartIndex, -1, numCells);
	set_array_to_value<int> << < gridSize, blockSize >> >(d_cellEndIndex, -1, numCells);

	gpuErrchk(cudaMemcpy(&particles[0], d_output, 3 * numParticles*sizeof(float), cudaMemcpyDeviceToHost));
	//gpuErrchk(cudaMemcpy(h_pos, d_pos, numParticles*sizeof(float4), cudaMemcpyDeviceToHost));
	
	/*for (int i = 0; i < particles.size()/3; i++){
		std::cout << "i: " << i << " " << particles[3*i] << " " << particles[3*i+1] << " " << particles[3*i+2] << std::endl;
	}*/
}






//void CudaDEMPhysicsEngine::uniformFiniteGridAlgorithm()
//{
//	dim3 gridSize(32, 1, 1);
//	dim3 blockSize(32, 1, 1);
//	build_spatial_grid<<< gridSize, blockSize >>>
//		(
//			d_pos, 
//			d_particleIndex, 
//			d_cellHash, 
//			numParticles, 
//			particleGridDim,
//			particleGridSize
//		);
//
//	thrust::device_ptr<int> t_a(d_cellHash);
//	thrust::device_ptr<int> t_b(d_particleIndex);
//	thrust::sort_by_key(t_a, t_a + numParticles, t_b);
//
//	reorder_particles<<< gridSize, blockSize >>>
//		(
//			d_pos,
//			d_spos,
//			d_vel,
//			d_svel,
//			d_particleType,
//			d_sparticleType,
//			d_cellStartIndex,
//			d_cellEndIndex,
//			d_cellHash,
//			d_particleIndex,
//			numParticles
//		);
//
//	 calculate_collisions<<< gridSize, blockSize >>>
//	 	(
//	 		d_spos,  
//			d_svel,
//			d_acc,
//	 		d_cellStartIndex,
//	 		d_cellEndIndex,
//	 		d_cellHash,
//	 		d_particleIndex,
//	 		numParticles,
//	 		particleGridDim
//	 	);
//
//	 update_particles2<<< gridSize, blockSize >>>
//	 	(
//	 		d_spos,
//	 		d_svel,
//	 		d_acc,
//	 		dt,
//	 		numParticles,
//	 		particleGridSize
//	 	);
//
//	copy_sph_arrays<<< gridSize, blockSize >>>
//		(
//			d_pos,
//			d_spos,
//			d_vel,
//			d_svel,
//			d_particleType,
//			d_sparticleType,
//			d_output,
//			numParticles
//		);
//
//	set_array_to_value<int><<< gridSize, blockSize >>>(d_cellStartIndex, -1, numCells);
//	set_array_to_value<int><<< gridSize, blockSize >>>(d_cellEndIndex, -1, numCells);
//
//	gpuErrchk(cudaMemcpy(&particles[0], d_output, 3 * numParticles*sizeof(float), cudaMemcpyDeviceToHost));
//	gpuErrchk(cudaMemcpy(h_pos, d_pos, numParticles*sizeof(float4), cudaMemcpyDeviceToHost));
//
//	/*for (int i = 0; i < 4 * 4; i++){
//		std::cout << h_pos[i].x << " " << h_pos[i].y << " " << h_pos[i].z << std::endl;
//	}*/
//	/*std::cout << h_pos[0].x << " " << h_pos[0].y << " " << h_pos[0].z << std::endl;
//	std::cout << h_pos[1].x << " " << h_pos[1].y << " " << h_pos[1].z << std::endl;
//	std::cout << h_pos[2].x << " " << h_pos[2].y << " " << h_pos[2].z << std::endl;
//	std::cout << h_pos[3].x << " " << h_pos[3].y << " " << h_pos[3].z << std::endl;*/
//}