#ifndef __PHYSICSSYSTEM_H__
#define __PHYSICSSYSTEM_H__

#include <vector>

#include <GL/glew.h>
#include <gl/gl.h>

#include "System.h"

#include "../core/Input.h"

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
			PhysicsSystem(std::vector<char> data);
			~PhysicsSystem();

			void* operator new(size_t size);
			void operator delete(void*);

			void init(World* world);
			void update(Input input);
	};
}

#endif