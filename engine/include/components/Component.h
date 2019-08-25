#ifndef __COMPONENT_H__
#define __COMPONENT_H__

#include "../core/common.h"

#include "../core/Guid.h"

namespace PhysicsEngine
{
	class Entity;
	class World;

	template <typename T>
	struct ComponentType { static const int type; };

	template <typename T>
	const int ComponentType<T>::type = -1;

	class Component
	{
		public:
			Guid componentId;
			Guid entityId;

		public:
			Component();
			virtual ~Component() = 0;

			virtual std::vector<char> serialize() = 0;
			virtual void deserialize(std::vector<char> data) = 0;

			Entity* getEntity(World* world);

			template<typename T>
			void latentDestroy(World* world)
			{
				world->latentDestroyComponent(entityId, componentId, getInstanceType<T>());
			}

			template<typename T>
			void immediateDestroy(World* world)
			{
				world->immediateDestroyComponent(entityId, componentId, getInstanceType<T>());
			}

			template<typename T>
			T* getComponent(World* world)
			{
				Entity* entity = getEntity(world);

				return entity->getComponent<T>(world);
			}
	};
}

#endif