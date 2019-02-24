#ifndef __BOIDSSYSTEM_H__
#define __BOIDSSYSTEM_H__

#include <vector>

#include <GL/glew.h>
#include <gl/gl.h>

#include "System.h"

#include "../core/Input.h"

#include "../cuda/BoidsDeviceData.cuh"

namespace PhysicsEngine
{
	class BoidsSystem : public System
	{
		private:
			std::vector<BoidsDeviceData> deviceData;

		public:
			BoidsSystem();
			BoidsSystem(std::vector<char> data);
			~BoidsSystem();

			void* operator new(size_t size);
			void operator delete(void*);

			void init(World* world);
			void update(Input input);
	};
}

#endif