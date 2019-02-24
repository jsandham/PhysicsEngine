#ifndef __CLOTHSYSTEM_H__
#define __CLOTHSYSTEM_H__

#include <vector>

#include <GL/glew.h>
#include <gl/gl.h>

#include "System.h"

#include "../core/Input.h"

#include "../cuda/ClothDeviceData.cuh"

namespace PhysicsEngine
{
	class ClothSystem : public System
	{
		private:
			std::vector<ClothDeviceData> deviceData;

		public:
			ClothSystem();
			ClothSystem(std::vector<char> data);
			~ClothSystem();

			void* operator new(size_t size);
			void operator delete(void*);

			void init(World* world);
			void update(Input input);
	};
}

#endif