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
			LineBuffer buffer;

			Material* colorMat;
			Shader* colorShader;

		public:
			DebugSystem();
			DebugSystem(std::vector<char> data);
			~DebugSystem();

			std::vector<char> serialize();
			void deserialize(std::vector<char> data);

			void init(World* world);
			void update(Input input);
	};
}

#endif