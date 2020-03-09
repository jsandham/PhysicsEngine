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
		bool doNotDestroy;
	};
#pragma pack(pop)
	
	class World;

	class Entity
	{
		private:
			Guid entityId;
		
		public:
			bool doNotDestroy;

		public:
			Entity();
			Entity(std::vector<char> data);
			~Entity();

			std::vector<char> serialize() const;
			std::vector<char> serialize(Guid entityId) const;
			void deserialize(std::vector<char> data);

			void latentDestroy(World* world);
			void immediateDestroy(World* world);

			template<typename T>
			T* addComponent(World* world)
			{
				return world->addComponent<T>(entityId);
			}

			template<typename T>
			T* addComponent(World* world, std::vector<char> data)
			{
				return world->addComponent<T>(data);
			}

			template<typename T>
			T* getComponent(World* world)
			{
				return world->getComponent<T>(entityId);
			}

			std::vector<std::pair<Guid, int>> getComponentsOnEntity(World* world);

			Guid getId() const;

		private:
			friend class World;
	};

	template <typename T>
	struct EntityType { static const int type; };

	template <typename T>
	const int EntityType<T>::type = -1;

	template <>
	const int EntityType<Entity>::type = 0;

	template <typename T>
	struct IsEntity { static bool value; };

	template <typename T>
	bool IsEntity<T>::value = false;

	template<>
	bool IsEntity<Entity>::value = true;
}

#endif