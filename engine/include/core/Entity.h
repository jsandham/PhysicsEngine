#ifndef __ENTITY_H__
#define __ENTITY_H__

#include "Guid.h"

namespace PhysicsEngine
{
// #pragma pack(push, 1)
	struct EntityData
	{
		Guid entityId;
	};
// #pragma pack(pop)
	
	class World;

	class Entity
	{
		private:
			World* world;

		public:
			Guid entityId;

		public:
			Entity();
			~Entity();

			void load(EntityData data);

			void setWorld(World* world);

			void latentDestroy();
			void immediateDestroy();
			Entity* instantiate();

			template<typename T>
			T* addComponent()
			{
				return world->addComponent<T>(entityId);
			}

			template<typename T>
			T* getComponent()
			{
				return world->getComponent<T>(entityId);
			}
	};
}

#endif