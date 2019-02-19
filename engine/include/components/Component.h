#ifndef __COMPONENT_H__
#define __COMPONENT_H__

#include <vector>

#include "../core/Guid.h"

namespace PhysicsEngine
{
	class Entity;
	class World;

	class Component
	{
		public:
			Guid componentId;
			Guid entityId;

		public:
			Component();
			virtual ~Component() = 0;

			Entity* getEntity(World* world);

			template<typename T>
			T* getComponent(World* world)
			{
				Entity* entity = getEntity(world);

				return entity->getComponent<T>(world);
			}

			template <typename T>
			static int getInstanceType()
			{
				// static variables only run the first time the function is called
			    static int id = nextValue();
			    return id;
			}

		private:
			static int nextValue()
			{
				// static variables only run the first time the function is called
			    static int id = 0;
			    int result = id;
			    ++id;
			    return result;
			}
	};
}

#endif