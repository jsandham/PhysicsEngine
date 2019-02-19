#ifndef __DEBUGSYSTEM_H__
#define __DEBUGSYSTEM_H__

#include <vector>

#include "System.h"

#include "../core/Input.h"
#include "../core/Material.h"
#include "../core/Shader.h"

namespace PhysicsEngine
{
	class DebugSystem : public System
	{
		private:
			Material* lineMaterial;
			Shader* lineShader;

		public:
			DebugSystem();
			DebugSystem(std::vector<char> data);
			~DebugSystem();

			void* operator new(size_t size);
			void operator delete(void*);

			void init(World* world);
			void update(Input input);
	};
}

#endif