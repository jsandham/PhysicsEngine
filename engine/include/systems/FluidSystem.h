#ifndef __FLUIDSYSTEM_H__
#define __FLUIDSYSTEM_H__

#include <vector>

#include <GL/glew.h>
#include <gl/gl.h>

#include "System.h"

#include "../core/Input.h"

#include "../cuda/FluidDeviceData.cuh"

namespace PhysicsEngine
{
	class FluidSystem : public System
	{
		private:
			std::vector<FluidDeviceData> deviceData;

		public:
			FluidSystem();
			FluidSystem(std::vector<char> data);
			~FluidSystem();

			std::vector<char> serialize();
			void deserialize(std::vector<char> data);

			void init(World* world);
			void update(Input input);
	};
}

#endif