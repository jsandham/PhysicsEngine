#ifndef __BOIDSDEVICEDATA_H__
#define __BOIDSDEVICEDATA_H__ 

#include <vector>

#include <vector_types.h>

#include <GL/glew.h>
#include <cuda.h>
// #include <cudagl.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>

#include "../core/Mesh.h"
// #include "../core/Material.h"
#include "../core/Shader.h"

#include "kernels/boids_kernels.cuh"

#define GLM_FORCE_RADIANS

#include "../glm/glm.hpp"
#include "../glm/gtx/normal.hpp"

namespace PhysicsEngine
{
	struct BoidsDeviceData
	{
		int numBoids;          // number of boids
		int numVoxels;         // number of voxels
		float h;
		int3 voxelGridDim;     // number of voxels in grid for x, y, and z directions 
		float3 voxelGridSize;  // size of grid for x, y, and z directions

		// used for timing
		float elapsedTime;
		cudaEvent_t start, stop;

		// pointers to host memory
		float4 *h_pos;
		float4 *h_vel;
		float4 *h_spos;
		float4 *h_svel;
		int *h_cellStartIndex;
		int *h_cellEndIndex;
		int *h_cellHash;
		int *h_boidsIndex;
		float *h_modelMatrices;

		// pointers to device memory
		float4 *d_pos;
		float4 *d_vel;
		float4 *d_spos;
		float4 *d_svel;
		float4 *d_scratch;
		int *d_cellStartIndex;
		int *d_cellEndIndex;
		int *d_cellHash;
		int *d_boidsIndex;
		float *d_modelMatrices;

		Mesh* mesh;
		Shader* shader;
		//Material* material;

		GLuint VAO;
		GLuint vertexVBO;
		GLuint normalVBO;
		GLuint texCoordVBO;
		GLuint instanceModelVBO;

		struct cudaGraphicsResource* cudaVertexVBO;
		struct cudaGraphicsResource* cudaNormalVBO;
	};

	void allocateBoidsDeviceData(BoidsDeviceData* boids);
	void deallocateBoidsDeviceData(BoidsDeviceData* boids);
	void initializeBoidsDeviceData(BoidsDeviceData* boids);
	void updateBoidsDeviceData(BoidsDeviceData* boids);

}

#endif