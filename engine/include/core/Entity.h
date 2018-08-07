#ifndef __ENTITY_H__
#define __ENTITY_H__

#include <vector>

namespace PhysicsEngine
{
	class Entity
	{
		private:
			int ind;

		public:
			int globalEntityIndex;
			int globalComponentIndices[8];
			int componentTypes[8];

			int entityId;
			int componentIds[8];

		public:
			Entity();
			~Entity();

			template<typename T>
			void addComponent(T* component)
			{
				component->globalEntityIndex = globalEntityIndex;

				componentTypes[ind] = Manager::getType<T>();
				globalComponentIndices[ind] = component->globalComponentIndex;

				ind++;
			}

			template<typename T>
			T* getComponent(std::vector<T*> components)
			{
				for (int i = 0; i < 8; i++){
					if (componentTypes[i] == Manager::getType<T>()){
						int globalComponentIndex = globalComponentIndices[i];

						return components[globalComponentIndex];
					}
				}

				return NULL;
			}
	};
}

#endif