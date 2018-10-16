#ifndef __PHYSICSSYSTEM_H__
#define __PHYSICSSYSTEM_H__

#include <GL/glew.h>
#include <gl/gl.h>

#include "System.h"

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
			std::vector<CudaSolid> cudaSolids;

			float timestep;
			float gravity;

			bool start = false;

		public:
			PhysicsSystem();
			PhysicsSystem(unsigned char* data);
			~PhysicsSystem();

			void init();
			void update();
	};
}

#endif