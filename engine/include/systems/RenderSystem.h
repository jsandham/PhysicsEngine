#ifndef __RENDERSYSTEM_H__
#define __RENDERSYSTEM_H__

#include <vector>

#include "System.h"

#include "../core/Input.h"
#include "../core/World.h"

#include "../graphics/ForwardRenderer.h"
#include "../graphics/GraphicsTargets.h"
#include "../graphics/GraphicsQuery.h"

namespace PhysicsEngine
{
	class RenderSystem : public System
	{
		private:
			ForwardRenderer mForwardRenderer;

		public:
			bool mRenderToScreen;

		public:
			RenderSystem();
			RenderSystem(std::vector<char> data);
			~RenderSystem();

			std::vector<char> serialize() const;
			std::vector<char> serialize(Guid systemId) const;
			void deserialize(std::vector<char> data);

			void init(World* world);
			void update(Input input);

			GraphicsTargets getGraphicsTargets() const;
			GraphicsQuery getGraphicsQuery() const;
	};

	template <>
	const int SystemType<RenderSystem>::type = 0;

	template< typename T>
	struct IsRenderSystem { static const bool value; };

	template<typename T>
	const bool IsRenderSystem<T>::value = false;

	template<>
	const bool IsRenderSystem<RenderSystem>::value = true;
	template<>
	const bool IsSystem<RenderSystem>::value = true;
}

#endif