#ifndef __ENTITY_H__
#define __ENTITY_H__

#include <vector>

#include "Guid.h"
#include "Types.h"

namespace PhysicsEngine
{
#pragma pack(push, 1)
	struct EntityHeader
	{
		Guid mEntityId;
		uint8_t mDoNotDestroy;
	};
#pragma pack(pop)
	
	class World;

	class Entity
	{
		private:
			Guid mEntityId;
		
		public:
			bool mDoNotDestroy;

		public:
			Entity();
			Entity(const std::vector<char>& data);
			~Entity();

			std::vector<char> serialize() const;
			std::vector<char> serialize(Guid entityId) const;
			void deserialize(const std::vector<char>& data);

			void latentDestroy(World* world);
			void immediateDestroy(World* world);

			template<typename T>
			T* addComponent(World* world)
			{
				return world->addComponent<T>(mEntityId);
			}

			template<typename T>
			T* addComponent(World* world, std::vector<char> data)
			{
				return world->addComponent<T>(data);
			}

			template<typename T>
			T* getComponent(World* world)
			{
				return world->getComponent<T>(mEntityId);
			}

			std::vector<std::pair<Guid, int>> getComponentsOnEntity(World* world);

			Guid getId() const;

		private:
			friend class World;
	};

	template <typename T>
	struct EntityType { static constexpr int type = PhysicsEngine::INVALID_TYPE;};
	template <typename T>
	struct IsEntity { static constexpr bool value = false; };
	template<typename>
	struct IsEntityInternal { static constexpr bool value = false; };

	template <>
	struct EntityType<Entity> { static constexpr int type = PhysicsEngine::ENTITY_TYPE;};
	template <>
	struct IsEntity<Entity> { static constexpr bool value = true; };
	template <>
	struct IsEntityInternal<Entity> { static constexpr bool value = true; };
}

#endif