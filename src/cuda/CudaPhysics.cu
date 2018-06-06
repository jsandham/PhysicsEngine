#include <iostream>
#include <random>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/unique.h>

#include "CudaPhysics.cuh"
#include "cloth_kernels.cuh"
#include "fluid_kernels.cuh"
#include "cuda_util.h"

#include "../core/Log.h"

using namespace PhysicsEngine;

void CudaPhysics::allocate(CudaCloth* cloth)
{
	int nx = cloth->nx;
	int ny = cloth->ny;

	// allocate memory on host
	cloth->h_pos = new float4[nx*ny];
	cloth->h_oldPos = new float4[nx*ny];
	cloth->h_acc = new float4[nx*ny];

	// allocate memory on device
	gpuErrchk(cudaMalloc((void**)&(cloth->d_pos), nx*ny*sizeof(float4)));
	gpuErrchk(cudaMalloc((void**)&(cloth->d_oldPos), nx*ny*sizeof(float4)));
	gpuErrchk(cudaMalloc((void**)&(cloth->d_acc), nx*ny*sizeof(float4)));
	gpuErrchk(cudaMalloc((void**)&(cloth->d_output), 3*nx*ny*sizeof(float)));
}

void CudaPhysics::deallocate(CudaCloth* cloth)
{
	// free memory on host
	delete[] cloth->h_pos;
	delete[] cloth->h_oldPos;
	delete[] cloth->h_acc;

	// free memory on device
	gpuErrchk(cudaFree(cloth->d_pos));
	gpuErrchk(cudaFree(cloth->d_oldPos));
	gpuErrchk(cudaFree(cloth->d_acc));
	gpuErrchk(cudaFree(cloth->d_output));
}

void CudaPhysics::initialize(CudaCloth* cloth)
{
	int nx = cloth->nx;
	int ny = cloth->ny;

	for (unsigned int i = 0; i < cloth->particles.size() / 3; i++){
		float4 hPos;
		hPos.x = cloth->particles[3 * i];
		hPos.y = cloth->particles[3 * i + 1];
		hPos.z = cloth->particles[3 * i + 2];
		hPos.w = 0.0f;

		float4 hOldPos = hPos;

		float4 hAcc;
		hAcc.x = 0.0f;
		hAcc.y = 0.0f;
		hAcc.z = 0.0f;
		hAcc.w = 0.0f;

		(cloth->h_pos)[i] = hPos;
		(cloth->h_oldPos)[i] = hOldPos;
		(cloth->h_acc)[i] = hAcc;
	}

	gpuErrchk(cudaMemcpy(cloth->d_pos, cloth->h_pos, nx*ny*sizeof(float4), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(cloth->d_oldPos, cloth->h_oldPos, nx*ny*sizeof(float4), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(cloth->d_acc, cloth->h_acc, nx*ny*sizeof(float4), cudaMemcpyHostToDevice));

	cloth->initCalled = true;
}

void CudaPhysics::update(CudaCloth* cloth)
{
	cudaGraphicsMapResources(1, &(cloth->vbo_cuda), 0);
	size_t num_bytes;

	cudaGraphicsResourceGetMappedPointer((void**)&(cloth->d_output), &num_bytes, cloth->vbo_cuda);

	//Log::Info("num bytes: %d", (int)num_bytes);

	dim3 blockSize(16, 16);
	dim3 gridSize(16, 16);

	for (int i = 0; i < 20; ++i)
	{
		// apply_constraints<<<gridSize, blockSize>>>
		// (
		// 	cloth->d_pos,
		// 	cloth->nx,
		// 	cloth->ny
		// );

		calculate_forces<<<gridSize, blockSize >>>
		(
			cloth->d_pos, 
			cloth->d_oldPos, 
			cloth->d_acc, 
			cloth->mass, 
			cloth->kappa, 
			cloth->c, 
			cloth->dt, 
			cloth->nx, 
			cloth->ny
		);

		verlet_integration<<<gridSize, blockSize>>>
		(
			cloth->d_pos, 
			cloth->d_oldPos, 
			cloth->d_acc, 
			cloth->d_output, 
			cloth->dt, 
			cloth->nx, 
			cloth->ny
		);
	}

	cudaGraphicsUnmapResources(1, &(cloth->vbo_cuda), 0);

	//int nx = cloth->nx;
	//int ny = cloth->ny;

	//gpuErrchk(cudaMemcpy(&((cloth->particles)[0]), cloth->d_output, 3*nx*ny*sizeof(float), cudaMemcpyDeviceToHost));
}





void CudaPhysics::allocate(CudaFluid* fluid)
{
	int numParticles = fluid->numParticles;
	int numCells = fluid->numCells;

	// allocate memory on host
	fluid->h_pos = new float4[numParticles];
	fluid->h_vel = new float4[numParticles];
	fluid->h_acc = new float4[numParticles];
	fluid->h_spos = new float4[numParticles];
	fluid->h_svel = new float4[numParticles];

	fluid->h_rho = new float[numParticles];
	fluid->h_rho0 = new float[numParticles];
	fluid->h_pres = new float[numParticles];
	//fluid->h_output = new float[3*numParticles];
	
	fluid->h_cellStartIndex = new int[numCells];
	fluid->h_cellEndIndex = new int[numCells];
	fluid->h_cellIndex = new int[numParticles];
	fluid->h_particleIndex = new int[numParticles];
	fluid->h_particleType = new int[numParticles];
	fluid->h_sparticleType = new int[numParticles];

	// allocate memory on device
	 gpuErrchk(cudaMalloc((void**)&(fluid->d_pos), numParticles*sizeof(float4)));
	 gpuErrchk(cudaMalloc((void**)&(fluid->d_vel), numParticles*sizeof(float4)));
	 gpuErrchk(cudaMalloc((void**)&(fluid->d_acc), numParticles*sizeof(float4)));
	 gpuErrchk(cudaMalloc((void**)&(fluid->d_spos), numParticles*sizeof(float4)));
	 gpuErrchk(cudaMalloc((void**)&(fluid->d_svel), numParticles*sizeof(float4)));

	 gpuErrchk(cudaMalloc((void**)&(fluid->d_rho), numParticles*sizeof(float)));
	 gpuErrchk(cudaMalloc((void**)&(fluid->d_rho0), numParticles*sizeof(float)));
	 gpuErrchk(cudaMalloc((void**)&(fluid->d_pres), numParticles*sizeof(float)));
	 gpuErrchk(cudaMalloc((void**)&(fluid->d_output), 3*numParticles*sizeof(float)));

	 gpuErrchk(cudaMalloc((void**)&(fluid->d_cellStartIndex), numCells*sizeof(int)));
	 gpuErrchk(cudaMalloc((void**)&(fluid->d_cellEndIndex), numCells*sizeof(int)));
	 gpuErrchk(cudaMalloc((void**)&(fluid->d_cellHash), numParticles*sizeof(int)));
	 gpuErrchk(cudaMalloc((void**)&(fluid->d_particleIndex), numParticles*sizeof(int)));
	 gpuErrchk(cudaMalloc((void**)&(fluid->d_particleType), numParticles*sizeof(int)));
	 gpuErrchk(cudaMalloc((void**)&(fluid->d_sparticleType), numParticles*sizeof(int)));
}

void CudaPhysics::deallocate(CudaFluid* fluid)
{
	// free memory on host
	delete [] fluid->h_pos;
	delete [] fluid->h_vel;
	delete [] fluid->h_acc;
	delete [] fluid->h_spos;
	delete [] fluid->h_svel;

	delete [] fluid->h_rho;
	delete [] fluid->h_rho0;
	delete [] fluid->h_pres;

	delete [] fluid->h_cellStartIndex;
	delete [] fluid->h_cellEndIndex;
	delete [] fluid->h_cellIndex;
	delete [] fluid->h_particleIndex;
	delete [] fluid->h_particleType;
	delete [] fluid->h_sparticleType;

	// free memory on device
	gpuErrchk(cudaFree(fluid->d_pos));
	gpuErrchk(cudaFree(fluid->d_vel));
	gpuErrchk(cudaFree(fluid->d_acc));
	gpuErrchk(cudaFree(fluid->d_spos));
	gpuErrchk(cudaFree(fluid->d_svel));

	gpuErrchk(cudaFree(fluid->d_rho));
	gpuErrchk(cudaFree(fluid->d_rho0));
	gpuErrchk(cudaFree(fluid->d_pres));
	gpuErrchk(cudaFree(fluid->d_output));

	gpuErrchk(cudaFree(fluid->d_cellStartIndex));
	gpuErrchk(cudaFree(fluid->d_cellEndIndex));
	gpuErrchk(cudaFree(fluid->d_cellHash));
	gpuErrchk(cudaFree(fluid->d_particleIndex));
	gpuErrchk(cudaFree(fluid->d_particleType));
	gpuErrchk(cudaFree(fluid->d_sparticleType));
}

void CudaPhysics::initialize(CudaFluid* fluid)
{
	// numParticles = particles.size() / 3;

	// //numFluidParticles = particles.size() / 3;

	// dt = 0.0075f;
	// kappa = 1.0f;
	// rho0 = 1000.0f;
	// mass = 0.01f;

	// h = grid->getDx();
	// h2 = h * h;
	// h6 = h2 * h2 * h2;
	// h9 = h6 * h2 * h;

	// numCells = grid->getNx() * grid->getNy() * grid->getNz();

	// particleGridDim.x = grid->getNx();
	// particleGridDim.y = grid->getNy();
	// particleGridDim.z = grid->getNz();

	// particleGridSize.x = grid->getX();
	// particleGridSize.y = grid->getY();
	// particleGridSize.z = grid->getZ();

	// allocateMemory();

	// for (int i = 0; i < numParticles; i++){
	// 	h_pos[i].x = particles[3 * i];
	// 	h_pos[i].y = particles[3 * i + 1];
	// 	h_pos[i].z = particles[3 * i + 2];
	// 	h_pos[i].w = 0.0f;

	// 	h_vel[i].x = 0.0f;
	// 	h_vel[i].y = 0.0f;
	// 	h_vel[i].z = 0.0f;
	// 	h_vel[i].w = 0.0f;

	// 	h_rho0[i] = rho0;

	// 	h_particleType[i] = particleTypes[i];
	// }
	
	// gpuErrchk(cudaMemcpy(d_pos, h_pos, numParticles*sizeof(float4), cudaMemcpyHostToDevice));
	// gpuErrchk(cudaMemcpy(d_vel, h_vel, numParticles*sizeof(float4), cudaMemcpyHostToDevice));
	// gpuErrchk(cudaMemcpy(d_rho0, h_rho0, numParticles*sizeof(float), cudaMemcpyHostToDevice));
	// gpuErrchk(cudaMemcpy(d_particleType, h_particleType, numParticles*sizeof(int), cudaMemcpyHostToDevice));

	// initCalled = true;
}

void CudaPhysics::update(CudaFluid* fluid)
{
	dim3 gridSize(256,1,1);
	dim3 blockSize(256,1,1);
	//set_array_to_value<int> <<< gridSize, blockSize >>>(fluid->d_cellStartIndex, -1, fluid->numCells);
	//set_array_to_value<int> <<< gridSize, blockSize >>>(fluid->d_cellEndIndex, -1, fluid->numCells);

	build_spatial_grid <<< gridSize, blockSize >>>
	(
		fluid->d_pos, 
		fluid->d_particleIndex, 
		fluid->d_cellHash, 
		fluid->numParticles, 
		fluid->particleGridDim,
		fluid->particleGridSize
	);

	thrust::device_ptr<int> t_a(fluid->d_cellHash);
	thrust::device_ptr<int> t_b(fluid->d_particleIndex);
	thrust::sort_by_key(t_a, t_a + fluid->numParticles, t_b);

	reorder_particles <<< gridSize, blockSize >>>
	(
		fluid->d_pos,
		fluid->d_spos,
		fluid->d_vel,
		fluid->d_svel,
		fluid->d_particleType,
		fluid->d_sparticleType,
		fluid->d_cellStartIndex,
		fluid->d_cellEndIndex,
		fluid->d_cellHash,
		fluid->d_particleIndex,
		fluid->numParticles
	);

	calculate_fluid_particle_density <<< gridSize, blockSize >>>
	(
		fluid->d_spos,  
		fluid->d_rho, 
		fluid->d_sparticleType,
		fluid->d_cellStartIndex,
		fluid->d_cellEndIndex,
		fluid->d_cellHash,
		fluid->d_particleIndex,
		fluid->numParticles,
		fluid->h2,
		fluid->h9,
		fluid->particleGridDim
	);

	calculate_solid_particle_density <<< gridSize, blockSize >>>
	(
		fluid->d_spos,
		fluid->d_rho,
		fluid->d_sparticleType,
		fluid->d_cellStartIndex,
		fluid->d_cellEndIndex,
		fluid->d_cellHash,
		fluid->d_particleIndex,
		fluid->numParticles,
		fluid->h2,
		fluid->h9,
		fluid->particleGridDim
	);

	calculate_pressure <<< gridSize, blockSize >>>
	(
		fluid->d_rho,
		fluid->d_rho0,
		fluid->d_pres,
		fluid->numParticles,
		fluid->kappa
	);

	apply_pressure_and_gravity_acceleration <<< gridSize, blockSize >>>
	(
		fluid->d_spos, 
		fluid->d_svel,
		fluid->d_rho,
		fluid->d_pres,
		fluid->d_sparticleType,
		fluid->d_cellStartIndex,
		fluid->d_cellEndIndex,
		fluid->d_cellHash,
		fluid->d_particleIndex,
		fluid->numParticles,
		fluid->dt,
		fluid->h,
		fluid->h6,
		fluid->particleGridDim
	);

	compute_solid_particle_velocity <<< gridSize, blockSize >>>
	(
		fluid->d_spos,
		fluid->d_svel,
		fluid->d_sparticleType,
		fluid->numParticles
	);

	apply_xsph_viscosity <<< gridSize, blockSize >>>
	(
		fluid->d_spos,
		fluid->d_svel,
		fluid->d_rho,
		fluid->d_sparticleType,
		fluid->d_cellStartIndex,
		fluid->d_cellEndIndex,
		fluid->d_cellHash,
		fluid->d_particleIndex,
		fluid->numParticles,
		fluid->dt,
		fluid->h,
		fluid->h6,
		fluid->particleGridDim
	);

	update_particles<<< gridSize, blockSize >>>
	(
		fluid->d_spos,
		fluid->d_svel,
		fluid->d_sparticleType,
		fluid->dt,
		fluid->h,
		fluid->numParticles,
		fluid->particleGridSize
	);

	copy_sph_arrays<<< gridSize, blockSize >>>
	(
		fluid->d_pos,
		fluid->d_spos,
		fluid->d_vel,
		fluid->d_svel,
		fluid->d_particleType,
		fluid->d_sparticleType,
		fluid->d_output,
		fluid->numParticles
	);

	gpuErrchk(cudaMemcpy(&((fluid->particles)[0]), fluid->d_output, 3*fluid->numParticles*sizeof(float), cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(fluid->h_pos, fluid->d_pos, fluid->numParticles*sizeof(float4), cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(fluid->h_rho, fluid->d_rho, fluid->numParticles*sizeof(float), cudaMemcpyDeviceToHost));
}