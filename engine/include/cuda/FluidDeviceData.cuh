#ifndef __FLUIDDEVICEDATA_H__
#define __FLUIDDEVICEDATA_H__ 

#include <vector>

#include <vector_types.h>

#include <cuda.h>
//#include <cudagl.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>

#include "kernels/fluid_kernels.cuh"

#define GLM_FORCE_RADIANS

#include "../glm/glm.hpp"
#include "../glm/gtx/normal.hpp"

namespace PhysicsEngine
{
	struct FluidDeviceData
	{
		float h, h2, h6, h9;      // grid parameters
		float dt;                 // TODO: remove. use Physics.time instead
		float kappa;              // kappa
		float rho0;               // particle rest density
		float mass;	              // particle mass
 
		int numParticles;         // number of particles
		int numCells;             // number of cells
		int3 particleGridDim;     // number of voxels in grid for x, y, and z directions 
		float3 particleGridSize;  // size of grid for x, y, and z directions
		//VoxelGrid *grid;

		std::vector<float> particles;
		std::vector<int> particleTypes;

		// used for timing
		float elapsedTime;
		cudaEvent_t start, stop;

		bool initCalled;

		// pointers to host memory
		float4 *h_pos;
		float4 *h_vel;
		float4 *h_acc;
		float4 *h_spos;
		float4 *h_svel;
		float *h_rho, *h_rho0, *h_pres;
		int *h_cellStartIndex;
		int *h_cellEndIndex;
		int *h_cellIndex;
		int *h_particleIndex;
		int *h_particleType;
		int *h_sparticleType;

		// pointers to device memory
		float4 *d_pos;
		float4 *d_vel;
		float4 *d_acc;
		float4 *d_spos;
		float4 *d_svel;
		float *d_rho, *d_rho0, *d_pres;
		float *d_output;
		int *d_cellStartIndex;
		int *d_cellEndIndex;
		int *d_cellHash;
		int *d_particleIndex;
		int *d_particleType;
		int *d_sparticleType;
	};

	void allocateFluidDeviceData(FluidDeviceData* fluid);
	void deallocateFluidDeviceData(FluidDeviceData* fluid);
	void initializeFluidDeviceData(FluidDeviceData* fluid);
	void updateFluidDeviceData(FluidDeviceData* fluid);

}

#endif