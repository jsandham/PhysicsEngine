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
			bool isActive;

			int entityId;
			int componentIds[8];

		public:
			Entity();
			~Entity();

			void setManager(Manager* manager);

			template<typename T>
			void addComponent()
			{
				// TODO....
			}

			template<typename T>
			T* getComponent()
			{
				return manager->getComponent<T>(entityId);
			}
	};
}

#endif