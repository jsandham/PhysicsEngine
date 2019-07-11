#ifndef __RENDERSYSTEM_H__
#define __RENDERSYSTEM_H__

#include <vector>

#include "System.h"

#include "../core/Input.h"
#include "../core/World.h"

#include "../graphics/ForwardRenderer.h"
#include "../graphics/DeferredRenderer.h"
#include "../graphics/DebugRenderer.h"

namespace PhysicsEngine
{
	class RenderSystem : public System
	{
		private:
			unsigned int pass;

			ForwardRenderer forwardRenderer;
			DeferredRenderer deferredRenderer;  
			DebugRenderer debugRenderer;


			Guid testId;

		public:
			RenderSystem();
			RenderSystem(std::vector<char> data);
			~RenderSystem();

			std::vector<char> serialize();
			void deserialize(std::vector<char> data);

			void init(World* world);
			void update(Input input);
	};
}

#endif