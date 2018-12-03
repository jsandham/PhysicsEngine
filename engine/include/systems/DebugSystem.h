#ifndef __DEBUGSYSTEM_H__
#define __DEBUGSYSTEM_H__

#include "System.h"

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
			DebugSystem(unsigned char* data);
			~DebugSystem();

			void init();
			void update();
	};
}

#endif