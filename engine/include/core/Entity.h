#ifndef __ENTITY_H__
#define __ENTITY_H__

namespace PhysicsEngine
{
	class Manager;

	class Entity
	{
		private:
			Manager* manager;

		public:
			int entityId;
			int componentIds[8];

		public:
			Entity();
			~Entity();

			void setManager(Manager* manager);

			void latentDestroy();
			void immediateDestroy();
			Entity* instantiate();

			template<typename T>
			void addComponent()
			{
				manager->addComponent<T>(entityId);
			}

			template<typename T>
			T* getComponent()
			{
				return manager->getComponent<T>(entityId);
			}
	};
}

#endif