#ifndef __RENDERSYSTEM_H__
#define __RENDERSYSTEM_H__

#include <vector>

#include "System.h"

#include "../core/Input.h"
#include "../core/World.h"

#include "../graphics/ForwardRenderer.h"
#include "../graphics/DeferredRenderer.h"
#include "../graphics/RenderObject.h"

namespace PhysicsEngine
{
	class RenderSystem : public System
	{
		private:
			ForwardRenderer mForwardRenderer;
			DeferredRenderer mDeferredRenderer;

			std::vector<RenderObject> mRenderObjects;

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
			void update(Input input, Time time);

		private:
			void registerRenderAssets(World* world);
			void registerCameras(World* world);
			void updateRenderObjects(World* world, std::vector<RenderObject>& renderObjects);
			void updateModelMatrices(World* world, std::vector<RenderObject>& renderObjects);
			void cullRenderObjects(Camera* camera, std::vector<RenderObject>& renderObjects);
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