#ifndef __COMPONENT_H__
#define __COMPONENT_H__

#include <vector>

#include "../core/Guid.h"

namespace PhysicsEngine
{
	class Entity;
	class Manager;

	class Component
	{
		private:
			Manager* manager;

		public:
			bool isActive;

			Guid componentId;
			Guid entityId;

		public:
			Component();
			virtual ~Component() = 0;

			void setManager(Manager* manager);

			Entity* getEntity();

			template<typename T>
			T* getComponent()
			{
				Entity* entity = getEntity();

				return entity->getComponent<T>();
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