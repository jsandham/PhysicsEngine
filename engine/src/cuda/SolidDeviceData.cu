#include <iostream>
#include <random>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/unique.h>

#include "../../include/cuda/SolidDeviceData.cuh"
#include "../../include/cuda/cuda_util.h"
#include "../../include/cuda/kernels/solid_kernels.cuh"

using namespace PhysicsEngine;
using namespace SolidKernels;

void PhysicsEngine::allocateSolidDeviceData(SolidDeviceData* solid)
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

	// allocate solver 
	gpuErrchk(cudaMalloc((void**)&((solid->jacobi).d_xnew), n*sizeof(float)));
	gpuErrchk(cudaMalloc((void**)&((solid->jacobi).d_diag), n*sizeof(float)));
}

void PhysicsEngine::deallocateSolidDeviceData(SolidDeviceData* solid)
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

	gpuErrchk(cudaFree(solid->d_rowA));
	gpuErrchk(cudaFree(solid->d_colA));
	gpuErrchk(cudaFree(solid->d_valA));

	// deallocate solver
	gpuErrchk(cudaFree((solid->jacobi).d_xnew));
	gpuErrchk(cudaFree((solid->jacobi).d_diag));
}

void PhysicsEngine::initializeSolidDeviceData(SolidDeviceData* solid)
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

	std::cout << "n: " << n << " ne: " << ne << " npe: " << npe << std::endl;

	// form global mass matrix;
	assembleCSR(solid->h_valA, solid->h_rowA, solid->h_colA, solid->h_connect, solid->h_localElementMatrices, n, ne, npe);

	for(int i = 0; i < 20; i++)
	{
		std::cout << "row: " << solid->h_rowA[i] << std::endl;
	}

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

void PhysicsEngine::updateSolidDeviceData(SolidDeviceData* solid)
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

void PhysicsEngine::assembleCSR(float* values, int* rowPtrs, int* columns, int* connect, float* localMatrices, int n, int ne, int npe)
{
	int MAX_NNZ = 32;

	// initialize rowPtr to zeros
	for(int i = 0; i < n + 1; i++){
		rowPtrs[i] = 0;
	}

	// initialize columns to -1
	for(int i = 0; i < MAX_NNZ*n; i++){
		columns[i] = -1;
	}

	// update global matrix values and columns arrays
	for(int k = 0; k < ne; k++){
	    for(int i = 0; i < npe; i++){
	      	for(int j = 0; j < npe; j++){
	        	int r = connect[npe*k + i];
	        	int c = connect[npe*k + j];
	        	float v = localMatrices[npe*npe*k + npe*i + j];
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