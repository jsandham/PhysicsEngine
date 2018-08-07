#ifndef __CUDACLOTHPhysicsEngine_CUH__
#define __CUDACLOTHPhysicsEngine_CUH_H__

#include <vector>
#include "vector_types.h"

#include "CudaPhysics.h"

using namespace PhysicsEngine;

class CudaClothPhysics : public CudaPhysics
{
	private:
		int nx, ny;

		std::vector<float> particles;
		std::vector<int> particleTypes;

		// host variables
		float4 *h_pos;
		float4 *h_oldPos;
		float4 *h_acc;

		// device variables
		float4 *d_pos;
		float4 *d_oldPos;
		float4 *d_acc;
		float *d_output;

		// used for timing
		float elapsedTime;
		cudaEvent_t start, stop;

		bool initCalled;

	public:
		float dt;
		float kappa;
		float c;
		float mass;

	public:
		CudaClothPhysics();
		~CudaClothPhysics();

		void init();
		void update();

		void setParticles(std::vector<float> &particles);
		void setParticleTypes(std::vector<int> &particleTypes);
		void setNx(int nx);
		void setNy(int ny);

		std::vector<float>& getParticles();
		std::vector<int>& getParticleTypes();
		int getNx();
		int getNy();

	private:
		void allocateMemory();
		void deallocateMemory();

		void provotAlgorithm();
};

#endif