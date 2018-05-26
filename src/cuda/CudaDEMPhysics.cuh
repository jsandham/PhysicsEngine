#ifndef __CUDADEMPARTICLEPhysicsEngine_CUH__
#define __CUDADEMPARTICLEPhysicsEngine_CUH__


#include <vector>

#include "vector_types.h"
#include "glm/glm.hpp"

#include "CudaPhysics.h"
#include "VoxelGrid.h"

using namespace PhysicsEngine;

class CudaDEMPhysics : public CudaPhysics
{
	private:
		// particle grid 
		int numParticles, numCells;
		int3 particleGridDim;
		float3 particleGridSize;
		VoxelGrid *grid;

		std::vector<float> particles;
		std::vector<int> particleTypes;

		// host variables
		float4 *h_pos;
		float4 *h_vel;
		float4 *h_oldPos;
		float4 *h_oldVel;
		float4 *h_spos;
		float4 *h_svel;
	
		int *h_cellStartIndex;
		int *h_cellEndIndex;
		int *h_cellIndex;
		int *h_particleIndex;
		int *h_particleType;
		int *h_sparticleType;

		// device variables
		float4 *d_pos;
		float4 *d_oldPos;
		float4 *d_vel;
		float4 *d_oldVel;
		float4 *d_spos;
		float4 *d_svel;
		float4 *d_soldPos;
		float4 *d_soldVel;
		float *d_output;
		int *d_cellStartIndex;
		int *d_cellEndIndex;
		int *d_cellHash;
		int *d_particleIndex;
		int *d_particleType;
		int *d_sparticleType;

		// used for timing
		float elapsedTime;
		cudaEvent_t start, stop;

		bool initCalled;

	public:
		float dt;

	public:
		CudaDEMPhysics();
		~CudaDEMPhysics();

		void init();
		void update();

		void setGridDomain(VoxelGrid *grid);
		void setParticles(std::vector<float> &particles);
		void setParticleTypes(std::vector<int> &particleTypes);
		std::vector<float>& getParticles();
		std::vector<int>& getParticleTypes();

	private:
		void allocateMemory();
		void deallocateMemory();
		void uniformFiniteGridAlgorithm();
};

#endif
