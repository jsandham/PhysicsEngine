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
#include "solid_kernels.cuh"
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

void CudaPhysics::deallocate(CudaCloth* cloth)
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

void CudaPhysics::update(CudaCloth* cloth)
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




void CudaPhysics::allocate(CudaSolid* solid)
{
	int ng = solid->ng;
	int n = solid->n;
	int npe = solid->npe;
	int ne = solid->ne;
	int ne_b = solid->ne_b;
	int npe_b = solid->npe_b;

	// allocate memory on host
	solid->h_pos = new float4[n];
	solid->h_oldPos = new float4[n];
	solid->h_acc = new float4[n];
	solid->h_connect = new int[npe*ne];
	solid->h_bconnect = new int[ne_b*(npe_b+1)];
	solid->h_groups = new int[ng];
	solid->h_triangleIndices = new int[ne_b*npe_b];
	solid->h_triangleVertices = new float[3*ne_b*npe_b];
	solid->h_triangleNormals = new float[3*ne_b*npe_b];
	solid->h_localElementMatrices = new float[ne*npe*npe];

	solid->h_rowA = new int[n + 1];
	solid->h_colA = new int[32*n];
	solid->h_valA = new float[32*n];

	// allocate memory on device
	gpuErrchk(cudaMalloc((void**)&(solid->d_pos), n*sizeof(float4)));
	gpuErrchk(cudaMalloc((void**)&(solid->d_oldPos), n*sizeof(float4)));
	gpuErrchk(cudaMalloc((void**)&(solid->d_acc), n*sizeof(float4)));
	gpuErrchk(cudaMalloc((void**)&(solid->d_connect), npe*ne*sizeof(int)));
	gpuErrchk(cudaMalloc((void**)&(solid->d_bconnect), ne_b*(npe_b+1)*sizeof(int)));
	gpuErrchk(cudaMalloc((void**)&(solid->d_groups), ng*sizeof(int)));
	gpuErrchk(cudaMalloc((void**)&(solid->d_triangleIndices), ne_b*npe_b*sizeof(int)));
	gpuErrchk(cudaMalloc((void**)&(solid->d_triangleVertices), 3*ne_b*npe_b*sizeof(float)));
	gpuErrchk(cudaMalloc((void**)&(solid->d_triangleNormals), 3*ne_b*npe_b*sizeof(float)));
	gpuErrchk(cudaMalloc((void**)&(solid->d_localElementMatrices), ne*npe*npe*sizeof(float)));
	
	gpuErrchk(cudaMalloc((void**)&(solid->d_rowA), (n+1)*sizeof(int)));
	gpuErrchk(cudaMalloc((void**)&(solid->d_colA), 32*n*sizeof(int)));
	gpuErrchk(cudaMalloc((void**)&(solid->d_valA), 32*n*sizeof(float)));
}

void CudaPhysics::deallocate(CudaSolid* solid)
{
	// allocate memory on host
	delete [] solid->h_pos; 
	delete [] solid->h_oldPos; 
	delete [] solid->h_acc; 
	delete [] solid->h_connect; 
	delete [] solid->h_bconnect; 
	delete [] solid->h_groups; 
	delete [] solid->h_triangleIndices; 
	delete [] solid->h_triangleVertices; 
	delete [] solid->h_triangleNormals; 
	delete [] solid->h_localElementMatrices;

	delete [] solid->h_rowA;
	delete [] solid->h_colA;
	delete [] solid->h_valA;

	// allocate memory on device
	gpuErrchk(cudaFree(solid->d_pos));
	gpuErrchk(cudaFree(solid->d_oldPos));
	gpuErrchk(cudaFree(solid->d_acc));
	gpuErrchk(cudaFree(solid->d_connect));
	gpuErrchk(cudaFree(solid->d_bconnect));
	gpuErrchk(cudaFree(solid->d_groups));
	gpuErrchk(cudaFree(solid->d_triangleIndices));
	gpuErrchk(cudaFree(solid->d_triangleVertices));
	gpuErrchk(cudaFree(solid->d_triangleNormals));
	gpuErrchk(cudaFree(solid->d_localElementMatrices));

	gpuErrchk(cudaFree(solid->h_rowA));
	gpuErrchk(cudaFree(solid->h_colA));
	gpuErrchk(cudaFree(solid->h_valA));
}

void CudaPhysics::initialize(CudaSolid* solid)
{
	int ng = solid->ng;
	int n = solid->n;
	int npe = solid->npe;
	int ne = solid->ne;
	int ne_b = solid->ne_b;
	int npe_b = solid->npe_b;
	//int type = solid->type;
	//int type_b = solid->type_b;

	for(unsigned int i = 0; i < solid->vertices.size() / 3; i++){
		float4 hPos;
		hPos.x = solid->vertices[3 * i];
		hPos.y = solid->vertices[3 * i + 1];
		hPos.z = solid->vertices[3 * i + 2];
		hPos.w = 0.0f;

		float4 hOldPos = hPos;

		float4 hAcc;
		hAcc.x = 0.0f;
		hAcc.y = 0.0f;
		hAcc.z = 0.0f;
		hAcc.w = 0.0f;

		(solid->h_pos)[i] = hPos;
		(solid->h_oldPos)[i] = hOldPos;
		(solid->h_acc)[i] = hAcc;
	}

	for(unsigned int i = 0; i < solid->connect.size(); i++){
		(solid->h_connect)[i] = (solid->connect)[i];
	}

	for(unsigned int i = 0; i < solid->bconnect.size(); i++){
		(solid->h_bconnect)[i] = (solid->bconnect)[i];
		if(i < 20){
			std::cout << (solid->h_bconnect)[i] << " " << std::endl;
		}
	}

	for(unsigned int i = 0; i < solid->groups.size(); i++){
		(solid->h_groups)[i] = (solid->groups)[i];
	}

	for(unsigned int i = 0; i < ne_b; i++){
		// TODO: this only works with linear triangular elements
		int ind1 = solid->h_bconnect[4*i + 1] - 1;
		int ind2 = solid->h_bconnect[4*i + 2] - 1;
		int ind3 = solid->h_bconnect[4*i + 3] - 1;

		solid->h_triangleIndices[3*i] = ind1;
		solid->h_triangleIndices[3*i + 1] = ind2;
		solid->h_triangleIndices[3*i + 2] = ind3;

		glm::vec3 a = glm::vec3(solid->vertices[3*ind1], solid->vertices[3*ind1 + 1], solid->vertices[3*ind1 + 2]);
		glm::vec3 b = glm::vec3(solid->vertices[3*ind2], solid->vertices[3*ind2 + 1], solid->vertices[3*ind2 + 2]);
		glm::vec3 c = glm::vec3(solid->vertices[3*ind3], solid->vertices[3*ind3 + 1], solid->vertices[3*ind3 + 2]);

		glm::vec3 normal = glm::triangleNormal(a, b, c);

		solid->h_triangleVertices[9*i] = a.x;
		solid->h_triangleVertices[9*i + 1] = a.y;
		solid->h_triangleVertices[9*i + 2] = a.z;
		solid->h_triangleVertices[9*i + 3] = b.x;
		solid->h_triangleVertices[9*i + 4] = b.y;
		solid->h_triangleVertices[9*i + 5] = b.z;
		solid->h_triangleVertices[9*i + 6] = c.x;
		solid->h_triangleVertices[9*i + 7] = c.y;
		solid->h_triangleVertices[9*i + 8] = c.z;

		solid->h_triangleNormals[9*i] = normal.x;
		solid->h_triangleNormals[9*i + 1] = normal.y;
		solid->h_triangleNormals[9*i + 2] = normal.z;
		solid->h_triangleNormals[9*i + 3] = normal.x;
		solid->h_triangleNormals[9*i + 4] = normal.y;
		solid->h_triangleNormals[9*i + 5] = normal.z;
		solid->h_triangleNormals[9*i + 6] = normal.x;
		solid->h_triangleNormals[9*i + 7] = normal.y;
		solid->h_triangleNormals[9*i + 8] = normal.z;
	}

	gpuErrchk(cudaMemcpy(solid->d_pos, solid->h_pos, n*sizeof(float4), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(solid->d_oldPos, solid->h_oldPos, n*sizeof(float4), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(solid->d_acc, solid->h_acc, n*sizeof(float4), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(solid->d_connect, solid->h_connect, npe*ne*sizeof(int), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(solid->d_bconnect, solid->h_bconnect, ne_b*(npe_b+1)*sizeof(int), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(solid->d_groups, solid->h_groups, ng*sizeof(int), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(solid->d_triangleIndices, solid->h_triangleIndices, ne_b*npe_b*sizeof(int), cudaMemcpyHostToDevice));

	size_t num_bytes;

	gpuErrchk(cudaGraphicsMapResources(1, &(solid->cudaVertexVBO), 0));
	gpuErrchk(cudaGraphicsResourceGetMappedPointer((void**)&(solid->d_triangleVertices), &num_bytes, solid->cudaVertexVBO));
	gpuErrchk(cudaMemcpy(solid->d_triangleVertices, solid->h_triangleVertices, 3*ne_b*npe_b*sizeof(float), cudaMemcpyHostToDevice));
	gpuErrchk(cudaGraphicsUnmapResources(1, &(solid->cudaVertexVBO), 0));

	gpuErrchk(cudaGraphicsMapResources(1, &(solid->cudaNormalVBO), 0));
	gpuErrchk(cudaGraphicsResourceGetMappedPointer((void**)&(solid->d_triangleNormals), &num_bytes, solid->cudaNormalVBO));
	gpuErrchk(cudaMemcpy(solid->d_triangleNormals, solid->h_triangleNormals, 3*ne_b*npe_b*sizeof(float), cudaMemcpyHostToDevice));
	gpuErrchk(cudaGraphicsUnmapResources(1, &(solid->cudaNormalVBO), 0));

	solid->initCalled = true;

	std::cout << " ne: " << solid->ne << " npe: " << solid->npe << " ne_b: " << solid->ne_b << " type: " << solid->type << std::endl;

	dim3 blockSize(64, 1);
	dim3 gridSize(64, 1);

	// find local mass matrices
	compute_local_mass_matrices<<<gridSize, blockSize>>>
	(
		solid->d_pos,
		solid->d_localElementMatrices,
		solid->d_connect,
		solid->ne,
		solid->npe,
		solid->type
	);

	gpuErrchk(cudaMemcpy(solid->h_localElementMatrices, solid->d_localElementMatrices, ne*npe*npe*sizeof(float), cudaMemcpyDeviceToHost));

	// form global mass matrix;
	assembleCSR(solid->h_valA, solid->h_rowA, solid->h_colA, solid->h_connect, solid->h_localElementMatrices, n, ne, npe);

	// find local stiffness matrices
	compute_local_stiffness_matrices<<<gridSize, blockSize>>>
	(
		solid->d_pos,
		solid->d_localElementMatrices,
		solid->d_connect,
		solid->ne,
		solid->npe,
		solid->type
	);

	gpuErrchk(cudaMemcpy(solid->h_localElementMatrices, solid->d_localElementMatrices, ne*npe*npe*sizeof(float), cudaMemcpyDeviceToHost));

	// form global stiffness matrix
	//assembleCSR(solid->h_valA, solid->h_rowA, solid->h_colA, solid->h_connect, solid->h_localElementMatrices, n, ne, npe);

	// gpuErrchk(cudaMemcpy(solid->h_localElementMatrices, solid->d_localElementMatrices, ne*npe*npe*sizeof(float), cudaMemcpyDeviceToHost));

	// for(int i = 0; i < 100; i++){
	// 	std::cout << "i: " << solid->h_localElementMatrices[i] << std::endl;
	// }
}

void CudaPhysics::update(CudaSolid* solid)
{
	std::cout << "BBBBBB" << std::endl;

	cudaGraphicsMapResources(1, &(solid->cudaVertexVBO), 0);
	size_t num_bytes;

	cudaGraphicsResourceGetMappedPointer((void**)&(solid->d_triangleVertices), &num_bytes, solid->cudaVertexVBO);

	// std::cout << " ne: " << solid->ne << " npe: " << solid->npe << " ne_b: " << solid->ne_b << " type: " << solid->type << std::endl;

	// dim3 blockSize(64, 1);
	// dim3 gridSize(64, 1);

	// for (int i = 0; i < 1; ++i)
	// {
	// 	compute_local_stiffness_matrices<<<gridSize, blockSize>>>
	// 	(
	// 		solid->d_pos,
	// 		solid->d_localStiffnessMatrices,
	// 		solid->d_connect,
	// 		solid->ne,
	// 		solid->npe,
	// 		solid->type
	// 	);
	// }

	cudaGraphicsUnmapResources(1, &(solid->cudaVertexVBO), 0);
}

void CudaPhysics::assembleCSR(float* values, int* rowPtrs, int* columns, int* connect, float* localMatrices, int n, int ne, int npe)
{
	int r, c = 0;
	float v;

	int MAX_NNZ = 32;

	// update global matrix values and columns arrays
	for(int k = 0; k < ne; k++){
	    for(int i = 0; i < npe; i++){
	      	for(int j = 0; j < npe; j++){
	        	r = connect[npe*k + i];
	        	c = connect[npe*k + j];
	        	v = localMatrices[npe*npe*k + npe*i + j];
	        	for(int p = MAX_NNZ*r - MAX_NNZ; p < MAX_NNZ*r; p++){
	          		if(columns[p] == -1){
	            		columns[p] = c - 1;
	            		values[p] = v;
	            		break;
	          		}
	          		else if(columns[p] == c - 1){
	            		values[p] += v;
	            		break;
	          		}
	        	}
	      	}
	    }
	}

  	//update row array
  	int jj = 0;
  	for(int i = 1; i < n + 1; i++){
    	for(int j = 0; j < MAX_NNZ; j++){
      		if(columns[i*MAX_NNZ-MAX_NNZ + j] == -1){jj = j; break;}
    	}
    	rowPtrs[i] = rowPtrs[i-1] + jj;
  	}

  	//sort (insertion) col and A arrays
  	for(int p = 0; p < n; p++){
    	for(int i = 0; i < rowPtrs[p+1] - rowPtrs[p]; i++){
      		int entry = columns[i + p*MAX_NNZ];
      		float entryA = values[i + p*MAX_NNZ];
      		int index = i + p*MAX_NNZ;
      		for(int j = i-1; j >= 0;j--){
        		if(entry < columns[j + p*MAX_NNZ]){
          			int a = columns[j + p*MAX_NNZ];
          			float b = values[j + p*MAX_NNZ];
          			columns[j + p*MAX_NNZ] = entry;
          			values[j + p*MAX_NNZ] = entryA;
          			columns[index] = a;
          			values[index] = b;
          			index = j + p*MAX_NNZ;
        		}
      		}
    	}
  	}

  	//compress col and A arrays
  	int index = 0;
  	for(int i = 0; i < n*MAX_NNZ; i++){
    	if(columns[i] == -1){ continue; }
    	columns[index] = columns[i];
    	values[index] = values[i];
    	index++;
  	}

  	//set unused parts of arrays to 0 or -1
  	for(int i = index; i < n*MAX_NNZ; i++){
    	columns[i] = -1;
    	values[i] = 0.0;
  	}
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