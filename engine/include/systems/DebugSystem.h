#ifndef __DEBUGSYSTEM_H__
#define __DEBUGSYSTEM_H__

#include <vector>

#include "System.h"

#include "../core/Input.h"
#include "../core/Material.h"
#include "../core/Shader.h"
#include "../graphics/Graphics.h"

namespace PhysicsEngine
{
	class DebugSystem : public System
	{
		private:
			Shader* fontShader;

			LineBuffer buffer;

		public:
			DebugSystem();
			DebugSystem(std::vector<char> data);
			~DebugSystem();

			void init(World* world);
			void update(Input input);
	};
}

#endif