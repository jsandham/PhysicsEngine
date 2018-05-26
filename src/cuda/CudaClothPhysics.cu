#include <iostream>
#include <stdio.h>
#include <cuda.h>

#include "CudaClothPhysics.cuh"

#include "kernels.cuh"
#include "cuda_util.cuh"

CudaClothPhysics::CudaClothPhysics()
{
	dt = 0.0015f;
	kappa = 100000.0f;
	c = 1.0f;
	mass = 1.0f;

	initCalled = false;
}

CudaClothPhysics::~CudaClothPhysics()
{
	if (initCalled){
		deallocateMemory();
	}
}

void CudaClothPhysics::allocateMemory()
{
	// allocate memory on host
	h_pos = new float4[nx*ny];
	h_oldPos = new float4[nx*ny];
	h_acc = new float4[nx*ny];

	// allocate memory on device
	gpuErrchk(cudaMalloc((void**)&d_pos, nx*ny*sizeof(float4)));
	gpuErrchk(cudaMalloc((void**)&d_oldPos, nx*ny*sizeof(float4)));
	gpuErrchk(cudaMalloc((void**)&d_acc, nx*ny*sizeof(float4)));
	gpuErrchk(cudaMalloc((void**)&d_output, 3*nx*ny*sizeof(float)));
}

void CudaClothPhysics::deallocateMemory()
{
	// free memory on host
	delete[] h_pos;
	delete[] h_oldPos;
	delete[] h_acc;

	// free memory on device
	gpuErrchk(cudaFree(d_pos));
	gpuErrchk(cudaFree(d_oldPos));
	gpuErrchk(cudaFree(d_acc));
	gpuErrchk(cudaFree(d_output));
}

void CudaClothPhysics::provotAlgorithm()
{
	dim3 blockSize(16, 16);
	dim3 gridSize(16, 16);

	calculate_forces<<<gridSize, blockSize >>>(d_pos, d_oldPos, d_acc, mass, kappa, c, dt, nx, ny);

	verlet_integration<<<gridSize, blockSize>>>(d_pos, d_oldPos, d_acc, d_output, dt, nx, ny);

	gpuErrchk(cudaMemcpy(&particles[0], d_output, 3*nx*ny*sizeof(float), cudaMemcpyDeviceToHost));
}

void CudaClothPhysics::init()
{
	std::cout << "particles.size(): " << particles.size() << " nx: " << nx << " ny: " << ny << std::endl;
	if (particles.size() != 3*nx*ny || !(nx % 2 == 0 && ny % 2 == 0) || !(nx != 0 && ny != 0)){
		std::cout << "CudaClothPhysicsEngine: Must set particles before calling init and m and n must be powers of 2." << std::endl;
		return;
	}

	allocateMemory();

	for (int i = 0; i < particles.size() / 3; i++){
		h_pos[i].x = particles[3 * i];
		h_pos[i].y = particles[3 * i + 1];
		h_pos[i].z = particles[3 * i + 2];
		h_pos[i].w = 0.0f;

		h_oldPos[i].x = particles[3 * i];
		h_oldPos[i].y = particles[3 * i + 1];
		h_oldPos[i].z = particles[3 * i + 2];
		h_oldPos[i].w = 0.0f;

		h_acc[i].x = 0.0f;
		h_acc[i].y = 0.0f;
		h_acc[i].z = 0.0f;
		h_acc[i].w = 0.0f;
	}

	gpuErrchk(cudaMemcpy(d_pos, h_pos, nx*ny*sizeof(float4), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_oldPos, h_oldPos, nx*ny*sizeof(float4), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_acc, h_acc, nx*ny*sizeof(float4), cudaMemcpyHostToDevice));

	initCalled = true;
}

void CudaClothPhysics::update()
{
	elapsedTime = 0;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	provotAlgorithm();

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
}

void CudaClothPhysics::setParticles(std::vector<float> &particles)
{
	this->particles = particles;
}

void CudaClothPhysics::setParticleTypes(std::vector<int> &particleTypes)
{
	this->particleTypes = particleTypes;
}

void CudaClothPhysics::setNx(int nx)
{
	this->nx = nx;
}

void CudaClothPhysics::setNy(int ny)
{
	this->ny = ny;
}

std::vector<float>& CudaClothPhysics::getParticles()
{
	return particles;
}

std::vector<int>& CudaClothPhysics::getParticleTypes()
{
	return particleTypes;
}

int CudaClothPhysics::getNx()
{
	return nx;
}

int CudaClothPhysics::getNy()
{
	return ny;
}