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

			std::vector<char> serialize();
			void deserialize(std::vector<char> data);

			void init(World* world);
			void update(Input input);
	};
}

#endif