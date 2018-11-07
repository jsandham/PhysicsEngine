#ifndef __ENTITY_H__
#define __ENTITY_H__

#include "Guid.h"

namespace PhysicsEngine
{
	class Manager;

	class Entity
	{
		private:
			Manager* manager;

		public:
			Guid entityId;

		public:
			Entity();
			~Entity();

			void setManager(Manager* manager);

			void latentDestroy();
			void immediateDestroy();
			Entity* instantiate();

			template<typename T>
			T* addComponent()
			{
				return manager->addComponent<T>(entityId);
			}

			template<typename T>
			T* getComponent()
			{
				return manager->getComponent<T>(entityId);
			}
	};
}

#endif