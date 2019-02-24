#ifndef __SOLIDSYSTEM_H__
#define __SOLIDSYSTEM_H__

#include <vector>

#include <GL/glew.h>
#include <gl/gl.h>

#include "System.h"

#include "../core/Input.h"

#include "../cuda/SolidDeviceData.cuh"

namespace PhysicsEngine
{
	class SolidSystem : public System
	{
		private:
			std::vector<SolidDeviceData> deviceData;

		public:
			SolidSystem();
			SolidSystem(std::vector<char> data);
			~SolidSystem();

			void* operator new(size_t size);
			void operator delete(void*);

			void init(World* world);
			void update(Input input);
	};
}

#endif