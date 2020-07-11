#ifndef __COMPONENT_H__
#define __COMPONENT_H__

#include "../core/Guid.h"

namespace PhysicsEngine
{
#pragma pack(push, 1)
	struct ComponentHeader
	{
		uint32_t mType;
		uint32_t mStartPtr;
		uint32_t mSize;
		Guid mComponentId;
	};
#pragma pack(pop)

	class Entity;
	class World;

	class Component
	{
		protected:
			Guid mComponentId;
			Guid mEntityId;

		public:
			Component();
			virtual ~Component() = 0;

			virtual std::vector<char> serialize() const = 0;
			virtual std::vector<char> serialize(Guid componentId, Guid entityId) const = 0;
			virtual void deserialize(const std::vector<char>& data) = 0;

			Entity* getEntity(World* world) const;

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

			Guid getId() const;
			Guid getEntityId() const;

		private:
			friend class World;
	};

	template <typename T>
	struct ComponentType { static const int type; };

	template <typename T>
	const int ComponentType<T>::type = -1;

	template <typename T>
	struct IsComponent { static const bool value; };

	template <typename T>
	const bool IsComponent<T>::value = false;

	template<>
	const bool IsComponent<Component>::value = true;

	template<typename T>
	struct IsComponentInternal { static const bool value; };

	template<>
	const bool IsComponentInternal<Component>::value = false;
}

#endif