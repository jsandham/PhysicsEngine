#ifndef __BOIDSDEVICEDATA_H__
#define __BOIDSDEVICEDATA_H__ 

#include <vector>

#include <vector_types.h>

#include <cuda.h>
// #include <cudagl.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>

#include "kernels/boids_kernels.cuh"

#define GLM_FORCE_RADIANS

#include "../glm/glm.hpp"
#include "../glm/gtx/normal.hpp"

namespace PhysicsEngine
{
	struct BoidsDeviceData
	{
		float h;               // grid parameters
		float dt;              // TODO: remove. use Physics.time instead

		int numBoids;          // number of boids
		int numCells;          // number of cells
		int3 boidsGridDim;     // number of voxels in grid for x, y, and z directions 
		float3 boidsGridSize;  // size of grid for x, y, and z directions

		// used for timing
		float elapsedTime;
		cudaEvent_t start, stop;

		bool initCalled;

		// pointers to host memory
		float4 *h_pos;
		float4 *h_vel;
		float4 *h_spos;
		float4 *h_svel;
		int *h_cellStartIndex;
		int *h_cellEndIndex;
		int *h_cellIndex;
		// int *h_particleIndex;
		// int *h_particleType;
		// int *h_sparticleType;
		int *h_triangleIndices;
		float *h_triangleVertices;
		float *h_triangleNormals;

		// pointers to device memory
		float4 *d_pos;
		float4 *d_vel;
		float4 *d_spos;
		float4 *d_svel;
		int *d_cellStartIndex;
		int *d_cellEndIndex;
		int *d_cellHash;
		// int *d_particleIndex;
		// int *d_particleType;
		// int *d_sparticleType;
		int *d_triangleIndices;
		float *d_triangleVertices;
		float *d_triangleNormals;

		struct cudaGraphicsResource* cudaVertexVBO;
		struct cudaGraphicsResource* cudaNormalVBO;
	};

	void allocateBoidsDeviceData(BoidsDeviceData* boids);
	void deallocateBoidsDeviceData(BoidsDeviceData* boids);
	void initializeBoidsDeviceData(BoidsDeviceData* boids);
	void updateBoidsDeviceData(BoidsDeviceData* boids);

}

#endif