#ifndef __PHYSICSSYSTEM_H__
#define __PHYSICSSYSTEM_H__

#include "System.h"

#include "../memory/Manager.h"

#include "../components/ParticlePhysics.h"
#include "../components/ParticleMesh.h"
#include "../components/Collider.h"
#include "../components/Rigidbody.h"

#include "../cuda/CudaPhysics.cuh"

namespace PhysicsEngine
{
	class PhysicsSystem : public System
	{
		private:
			std::vector<Collider*> colliders;
			std::vector<Rigidbody*> rigidbodies;

			std::vector<CudaCloth> cudaCloths;
			std::vector<CudaFluid> cudaFluids;
			std::vector<CudaFEM> cudaFEMs;

			float timestep;
			float gravity;

			bool start = false;

		public:
			PhysicsSystem(Manager *manager);
			~PhysicsSystem();

			void init();
			void update();
	};
}

#endif