#include <iostream>
#include <random>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/unique.h>

#include "../../include/cuda/ClothDeviceData.cuh"
#include "../../include/cuda/cuda_util.h"
#include "../../include/cuda/kernels/cloth_kernels.cuh"

using namespace PhysicsEngine;
using namespace ClothKernels;

void PhysicsEngine::allocateClothDeviceData(ClothDeviceData* cloth)
{
	int nx = cloth->nx;
	int ny = cloth->ny;

	// allocate memory on host
	cloth->h_pos = new float4[nx*ny];
	cloth->h_oldPos = new float4[nx*ny];
	cloth->h_acc = new float4[nx*ny];
	cloth->h_triangleIndices = new int[3*2*(nx-1)*(ny-1)];
	cloth->h_triangleVertices = new float[9*2*(nx-1)*(ny-1)];
	cloth->h_triangleNormals = new float[9*2*(nx-1)*(ny-1)];

	// allocate memory on device
	gpuErrchk(cudaMalloc((void**)&(cloth->d_pos), nx*ny*sizeof(float4)));
	gpuErrchk(cudaMalloc((void**)&(cloth->d_oldPos), nx*ny*sizeof(float4)));
	gpuErrchk(cudaMalloc((void**)&(cloth->d_acc), nx*ny*sizeof(float4)));
	gpuErrchk(cudaMalloc((void**)&(cloth->d_triangleIndices), 3*2*(nx-1)*(ny-1)*sizeof(int)));
	gpuErrchk(cudaMalloc((void**)&(cloth->d_triangleVertices), 9*2*(nx-1)*(ny-1)*sizeof(float)));
	gpuErrchk(cudaMalloc((void**)&(cloth->d_triangleNormals), 9*2*(nx-1)*(ny-1)*sizeof(float)));
}

void PhysicsEngine::deallocateClothDeviceData(ClothDeviceData* cloth)
{
	// free memory on host
	delete[] cloth->h_pos;
	delete[] cloth->h_oldPos;
	delete[] cloth->h_acc;
	delete[] cloth->h_triangleIndices;
	delete[] cloth->h_triangleVertices;
	delete[] cloth->h_triangleNormals;

	// free memory on device
	gpuErrchk(cudaFree(cloth->d_pos));
	gpuErrchk(cudaFree(cloth->d_oldPos));
	gpuErrchk(cudaFree(cloth->d_acc));
	gpuErrchk(cudaFree(cloth->d_triangleIndices));
	gpuErrchk(cudaFree(cloth->d_triangleVertices));
	gpuErrchk(cudaFree(cloth->d_triangleNormals));
}

void PhysicsEngine::initializeClothDeviceData(ClothDeviceData* cloth)
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

	// set up triangle mesh indices
	int index = 0;
	int triCount = 0;
	while(triCount < (nx-1)*(ny-1)){
		if(((index + 1) % nx) != 0){
			cloth->h_triangleIndices[3*index] = index;
			cloth->h_triangleIndices[3*index + 1] = nx + 1 + index;
			cloth->h_triangleIndices[3*index + 2] = nx + index;
			cloth->h_triangleIndices[3*index + 3] = index;
			cloth->h_triangleIndices[3*index + 4] = index + 1;
			cloth->h_triangleIndices[3*index + 5] = nx + 1 + index;
			triCount++;
		}

		index++;
	}

	for(int i = 0; i < 2*(nx-1)*(ny-1); i++){
		int ind1 = cloth->h_triangleIndices[3*i];
		int ind2 = cloth->h_triangleIndices[3*i + 1];
		int ind3 = cloth->h_triangleIndices[3*i + 2];

		glm::vec3 a = glm::vec3(cloth->particles[3*ind1], cloth->particles[3*ind1 + 1], cloth->particles[3*ind1 + 2]);
		glm::vec3 b = glm::vec3(cloth->particles[3*ind2], cloth->particles[3*ind2 + 1], cloth->particles[3*ind2 + 2]);
		glm::vec3 c = glm::vec3(cloth->particles[3*ind3], cloth->particles[3*ind3 + 1], cloth->particles[3*ind3 + 2]);

		glm::vec3 normal = glm::triangleNormal(a, b, c);

		cloth->h_triangleVertices[9*i] = a.x;
		cloth->h_triangleVertices[9*i + 1] = a.y;
		cloth->h_triangleVertices[9*i + 2] = a.z;
		cloth->h_triangleVertices[9*i + 3] = b.x;
		cloth->h_triangleVertices[9*i + 4] = b.y;
		cloth->h_triangleVertices[9*i + 5] = b.z;
		cloth->h_triangleVertices[9*i + 6] = c.x;
		cloth->h_triangleVertices[9*i + 7] = c.y;
		cloth->h_triangleVertices[9*i + 8] = c.z;

		cloth->h_triangleNormals[9*i] = normal.x;
		cloth->h_triangleNormals[9*i + 1] = normal.y;
		cloth->h_triangleNormals[9*i + 2] = normal.z;
		cloth->h_triangleNormals[9*i + 3] = normal.x;
		cloth->h_triangleNormals[9*i + 4] = normal.y;
		cloth->h_triangleNormals[9*i + 5] = normal.z;
		cloth->h_triangleNormals[9*i + 6] = normal.x;
		cloth->h_triangleNormals[9*i + 7] = normal.y;
		cloth->h_triangleNormals[9*i + 8] = normal.z;
	}

	gpuErrchk(cudaMemcpy(cloth->d_pos, cloth->h_pos, nx*ny*sizeof(float4), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(cloth->d_oldPos, cloth->h_oldPos, nx*ny*sizeof(float4), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(cloth->d_acc, cloth->h_acc, nx*ny*sizeof(float4), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(cloth->d_triangleIndices, cloth->h_triangleIndices, 3*2*(nx-1)*(ny-1)*sizeof(int), cudaMemcpyHostToDevice));

	size_t num_bytes;

	gpuErrchk(cudaGraphicsMapResources(1, &(cloth->cudaVertexVBO), 0));
	gpuErrchk(cudaGraphicsResourceGetMappedPointer((void**)&(cloth->d_triangleVertices), &num_bytes, cloth->cudaVertexVBO));
	gpuErrchk(cudaMemcpy(cloth->d_triangleVertices, cloth->h_triangleVertices, 9*2*(nx-1)*(ny-1)*sizeof(float), cudaMemcpyHostToDevice));
	gpuErrchk(cudaGraphicsUnmapResources(1, &(cloth->cudaVertexVBO), 0));

	gpuErrchk(cudaGraphicsMapResources(1, &(cloth->cudaNormalVBO), 0));
	gpuErrchk(cudaGraphicsResourceGetMappedPointer((void**)&(cloth->d_triangleNormals), &num_bytes, cloth->cudaNormalVBO));
	gpuErrchk(cudaMemcpy(cloth->d_triangleNormals, cloth->h_triangleNormals, 9*2*(nx-1)*(ny-1)*sizeof(float), cudaMemcpyHostToDevice));
	gpuErrchk(cudaGraphicsUnmapResources(1, &(cloth->cudaNormalVBO), 0));

	cloth->initCalled = true;
}

void PhysicsEngine::updateClothDeviceData(ClothDeviceData* cloth)
{
	gpuErrchk(cudaGraphicsMapResources(1, &(cloth->cudaVertexVBO), 0));
	size_t num_bytes;

	gpuErrchk(cudaGraphicsResourceGetMappedPointer((void**)&(cloth->d_triangleVertices), &num_bytes, cloth->cudaVertexVBO));

	dim3 blockSize(16, 16);
	dim3 gridSize(16, 16);

	for (int i = 0; i < 20; ++i)
	{
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
			cloth->dt, 
			cloth->nx, 
			cloth->ny
		);
	}

	update_triangle_mesh<<<gridSize, blockSize>>>
		(
			cloth->d_pos, 
			cloth->d_triangleIndices,
			cloth->d_triangleVertices,
			cloth->nx, 
			cloth->ny
		);

	gpuErrchk(cudaGraphicsUnmapResources(1, &(cloth->cudaVertexVBO), 0));
}