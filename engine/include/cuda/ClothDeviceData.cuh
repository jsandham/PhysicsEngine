#ifndef __CLOTHDEVICEDATA_H__
#define __CLOTHDEVICEDATA_H__ 

#include <vector>

#include <vector_types.h>

#include <cuda.h>
//#include <cudagl.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>

#include "kernels/cloth_kernels.cuh"

#define GLM_FORCE_RADIANS

#include "../glm/glm.hpp"
#include "../glm/gtx/normal.hpp"

namespace PhysicsEngine
{
	struct ClothDeviceData
	{
		int nx, ny;

		float dt;
		float kappa;            // spring stiffness coefficient
		float c;                // spring dampening coefficient
		float mass;             // mass

		std::vector<float> particles;
		std::vector<int> particleTypes;

		// used for timing
		float elapsedTime;
		cudaEvent_t start, stop;

		bool initCalled;

		// pointers to host memory
		float4 *h_pos;
		float4 *h_oldPos;
		float4 *h_acc;
		int *h_triangleIndices;
		float *h_triangleVertices;
		float *h_triangleNormals;

		// pointers to device memory
		float4 *d_pos;
		float4 *d_oldPos;
		float4 *d_acc;
		int *d_triangleIndices;
		float *d_triangleVertices;
		float *d_triangleNormals;

		struct cudaGraphicsResource* cudaVertexVBO;
		struct cudaGraphicsResource* cudaNormalVBO;
	};

	void allocateClothDeviceData(ClothDeviceData* cloth);
	void deallocateClothDeviceData(ClothDeviceData* cloth);
	void initializeClothDeviceData(ClothDeviceData* cloth);
	void updateClothDeviceData(ClothDeviceData* cloth);

}

#endif