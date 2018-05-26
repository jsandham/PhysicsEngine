#include <iostream>
#include <random>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/unique.h>

#include "CudaFluidPhysics.cuh"
#include "kernels.cuh"
#include "util_kernels.cuh"
#include "cuda_util.cuh"

#include <stdio.h>

#include <cuda.h>

#include "Util.h"

CudaFluidPhysics::CudaFluidPhysics()
{
	grid = NULL;

	initCalled = false;
}


//CudaFluidPhysics::CudaFluidPhysics(int numFluidParticles, int3 particleGridDim, float3 particleGridSize)
//{
//	this->numFluidParticles = numFluidParticles; 
//	this->particleGridDim = particleGridDim;
//	this->particleGridSize = particleGridSize;
//
//	dt = 0.0075f;
//	kappa = 1.0f;
//	rho0 = 1000.0f;
//	mass = 0.01f;
//	h = particleGridSize.x / particleGridDim.x;
//	h2 = h * h;
//	h6 = h2 * h2 * h2;
//	h9 = h6 * h2 * h;
//	numCells = particleGridDim.x * particleGridDim.y * particleGridDim.z;
//
//	// generate solid boundary particles
//	generateSolidParticles();
//
//	// allocate memory 
//	allocateMemory();
//}


CudaFluidPhysics::~CudaFluidPhysics()
{
	// free old memory 
	if (initCalled){
		deallocateMemory();
	}
}


void CudaFluidPhysics::allocateMemory()
{
	// allocate memory on host
	h_pos = new float4[numParticles];
	h_vel = new float4[numParticles];
	h_acc = new float4[numParticles];
	h_spos = new float4[numParticles];
	h_svel = new float4[numParticles];

	h_rho = new float[numParticles];
	h_rho0 = new float[numParticles];
	h_pres = new float[numParticles];
	//h_output = new float[3*numParticles];
	
	h_cellStartIndex = new int[numCells];
	h_cellEndIndex = new int[numCells];
	h_cellIndex = new int[numParticles];
	h_particleIndex = new int[numParticles];
	h_particleType = new int[numParticles];
	h_sparticleType = new int[numParticles];

	// allocate memory on device
	 gpuErrchk(cudaMalloc((void**)&d_pos, numParticles*sizeof(float4)));
	 gpuErrchk(cudaMalloc((void**)&d_vel, numParticles*sizeof(float4)));
	 gpuErrchk(cudaMalloc((void**)&d_acc, numParticles*sizeof(float4)));
	 gpuErrchk(cudaMalloc((void**)&d_spos, numParticles*sizeof(float4)));
	 gpuErrchk(cudaMalloc((void**)&d_svel, numParticles*sizeof(float4)));

	 gpuErrchk(cudaMalloc((void**)&d_rho, numParticles*sizeof(float)));
	 gpuErrchk(cudaMalloc((void**)&d_rho0, numParticles*sizeof(float)));
	 gpuErrchk(cudaMalloc((void**)&d_pres, numParticles*sizeof(float)));
	 gpuErrchk(cudaMalloc((void**)&d_output, 3*numParticles*sizeof(float)));

	 gpuErrchk(cudaMalloc((void**)&d_cellStartIndex, numCells*sizeof(int)));
	 gpuErrchk(cudaMalloc((void**)&d_cellEndIndex, numCells*sizeof(int)));
	 gpuErrchk(cudaMalloc((void**)&d_cellHash, numParticles*sizeof(int)));
	 gpuErrchk(cudaMalloc((void**)&d_particleIndex, numParticles*sizeof(int)));
	 gpuErrchk(cudaMalloc((void**)&d_particleType, numParticles*sizeof(int)));
	 gpuErrchk(cudaMalloc((void**)&d_sparticleType, numParticles*sizeof(int)));
}

void CudaFluidPhysics::deallocateMemory()
{
	// free memory on host
	delete [] h_pos;
	delete [] h_vel;
	delete [] h_acc;
	delete [] h_spos;
	delete [] h_svel;

	delete [] h_rho;
	delete [] h_rho0;
	delete [] h_pres;

	delete [] h_cellStartIndex;
	delete [] h_cellEndIndex;
	delete [] h_cellIndex;
	delete [] h_particleIndex;
	delete [] h_particleType;
	delete [] h_sparticleType;

	// free memory on device
	gpuErrchk(cudaFree(d_pos));
	gpuErrchk(cudaFree(d_vel));
	gpuErrchk(cudaFree(d_acc));
	gpuErrchk(cudaFree(d_spos));
	gpuErrchk(cudaFree(d_svel));

	gpuErrchk(cudaFree(d_rho));
	gpuErrchk(cudaFree(d_rho0));
	gpuErrchk(cudaFree(d_pres));
	gpuErrchk(cudaFree(d_output));

	gpuErrchk(cudaFree(d_cellStartIndex));
	gpuErrchk(cudaFree(d_cellEndIndex));
	gpuErrchk(cudaFree(d_cellHash));
	gpuErrchk(cudaFree(d_particleIndex));
	gpuErrchk(cudaFree(d_particleType));
	gpuErrchk(cudaFree(d_sparticleType));
}

//void CudaFluidPhysics::generateSolidParticles()
//{
//	Util::poissonSampler(solidParticles, 0.0f, 0.0f, 0.0f, particleGridSize.x, h, particleGridSize.z, h, 0.5*h, 30);
//	Util::poissonSampler(solidParticles, 0.0, particleGridSize.y - h, 0.0, particleGridSize.x, particleGridSize.y, particleGridSize.z, h, 0.5*h, 30);
//	Util::poissonSampler(solidParticles, 0.0, h, 0.0, h, particleGridSize.y - h, particleGridSize.z, h, 0.5*h, 30);
//	Util::poissonSampler(solidParticles, particleGridSize.x - h, h, 0.0, particleGridSize.x, particleGridSize.y - h, particleGridSize.z, h, 0.5*h, 30);
//	Util::poissonSampler(solidParticles, h, h, 0.0, particleGridSize.x - h, particleGridSize.y - h, h, h, 0.5*h, 30);
//}

void CudaFluidPhysics::init()
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

	//numFluidParticles = particles.size() / 3;

	dt = 0.0075f;
	kappa = 1.0f;
	rho0 = 1000.0f;
	mass = 0.01f;

	h = grid->getDx();
	h2 = h * h;
	h6 = h2 * h2 * h2;
	h9 = h6 * h2 * h;

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
		h_pos[i].x = particles[3 * i];
		h_pos[i].y = particles[3 * i + 1];
		h_pos[i].z = particles[3 * i + 2];
		h_pos[i].w = 0.0f;

		h_vel[i].x = 0.0f;
		h_vel[i].y = 0.0f;
		h_vel[i].z = 0.0f;
		h_vel[i].w = 0.0f;

		h_rho0[i] = rho0;

		h_particleType[i] = particleTypes[i];
	}
	
	gpuErrchk(cudaMemcpy(d_pos, h_pos, numParticles*sizeof(float4), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_vel, h_vel, numParticles*sizeof(float4), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_rho0, h_rho0, numParticles*sizeof(float), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_particleType, h_particleType, numParticles*sizeof(int), cudaMemcpyHostToDevice));

	initCalled = true;
}


void CudaFluidPhysics::update()
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

void CudaFluidPhysics::setGridDomain(VoxelGrid *grid)
{
	this->grid = grid;
}

void CudaFluidPhysics::setParticles(std::vector<float> &particles)
{
	this->particles = particles;
}

void CudaFluidPhysics::setParticleTypes(std::vector<int> &particleTypes)
{
	this->particleTypes = particleTypes;
}

std::vector<float>& CudaFluidPhysics::getParticles()
{
	return particles;
}

std::vector<int>& CudaFluidPhysics::getParticleTypes()
{
	return particleTypes;
}

void CudaFluidPhysics::uniformFiniteGridAlgorithm()
{
	dim3 gridSize(256,1,1);
	dim3 blockSize(256,1,1);
	set_array_to_value<int> <<< gridSize, blockSize >>>(d_cellStartIndex, -1, numCells);
	set_array_to_value<int> <<< gridSize, blockSize >>>(d_cellEndIndex, -1, numCells);

	build_spatial_grid <<< gridSize, blockSize >>>
		(
			d_pos, 
			d_particleIndex, 
			d_cellHash, 
			numParticles, 
			particleGridDim,
			particleGridSize
		);

	thrust::device_ptr<int> t_a(d_cellHash);
	thrust::device_ptr<int> t_b(d_particleIndex);
	thrust::sort_by_key(t_a, t_a + numParticles, t_b);

	reorder_particles <<< gridSize, blockSize >>>
		(
			d_pos,
			d_spos,
			d_vel,
			d_svel,
			d_particleType,
			d_sparticleType,
			d_cellStartIndex,
			d_cellEndIndex,
			d_cellHash,
			d_particleIndex,
			numParticles
		);

	calculate_fluid_particle_density <<< gridSize, blockSize >>>
		(
			d_spos,  
			d_rho, 
			d_sparticleType,
			d_cellStartIndex,
			d_cellEndIndex,
			d_cellHash,
			d_particleIndex,
			numParticles,
			h2,
			h9,
			particleGridDim
		);

	calculate_solid_particle_density <<< gridSize, blockSize >>>
		(
			d_spos,
			d_rho,
			d_sparticleType,
			d_cellStartIndex,
			d_cellEndIndex,
			d_cellHash,
			d_particleIndex,
			numParticles,
			h2,
			h9,
			particleGridDim
		);

	calculate_pressure <<< gridSize, blockSize >>>
		(
			d_rho,
			d_rho0,
			d_pres,
			numParticles,
			kappa
		);

	apply_pressure_and_gravity_acceleration <<< gridSize, blockSize >>>
		(
			d_spos, 
			d_svel,
			d_rho,
			d_pres,
			d_sparticleType,
			d_cellStartIndex,
			d_cellEndIndex,
			d_cellHash,
			d_particleIndex,
			numParticles,
			dt,
			h,
			h6,
			particleGridDim
		);

	compute_solid_particle_velocity <<< gridSize, blockSize >>>
		(
			d_spos,
			d_svel,
			d_sparticleType,
			numParticles
		);

	apply_xsph_viscosity <<< gridSize, blockSize >>>
		(
			d_spos,
			d_svel,
			d_rho,
			d_sparticleType,
			d_cellStartIndex,
			d_cellEndIndex,
			d_cellHash,
			d_particleIndex,
			numParticles,
			dt,
			h,
			h6,
			particleGridDim
		);

	update_particles<<< gridSize, blockSize >>>
		(
			d_spos,
			d_svel,
			d_sparticleType,
			dt,
			h,
			numParticles,
			particleGridSize
		);

	copy_sph_arrays<<< gridSize, blockSize >>>
		(
			d_pos,
			d_spos,
			d_vel,
			d_svel,
			d_particleType,
			d_sparticleType,
			d_output,
			numParticles
		);

	gpuErrchk(cudaMemcpy(&particles[0], d_output, 3*numParticles*sizeof(float), cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(h_pos, d_pos, numParticles*sizeof(float4), cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(h_rho, d_rho, numParticles*sizeof(float), cudaMemcpyDeviceToHost));
}