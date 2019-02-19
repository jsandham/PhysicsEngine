#ifndef __ENTITY_H__
#define __ENTITY_H__

#include <vector>

#include "Guid.h"

namespace PhysicsEngine
{
#pragma pack(push, 1)
	struct EntityHeader
	{
		Guid entityId;
	};
#pragma pack(pop)
	
	class World;

	class Entity
	{
		public:
			Guid entityId;

		public:
			Entity();
			Entity(std::vector<char> data);
			~Entity();

			void* operator new(size_t size);
			void operator delete(void*);

			void latentDestroy(World* world);
			void immediateDestroy(World* world);
			Entity* instantiate(World* world);

			template<typename T>
			T* addComponent(World* world)
			{
				return world->addComponent<T>(entityId);
			}

			template<typename T>
			T* getComponent(World* world)
			{
				return world->getComponent<T>(entityId);
			}
	};
}

#endif