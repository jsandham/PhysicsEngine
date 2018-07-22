#ifndef __ENTITY_H__
#define __ENTITY_H__

#include <iostream>
#include <typeindex>
#include <map>
#include <vector>
#include <string>

namespace PhysicsEngine
{
	class Component;

	class Entity
	{
		private:
			//int entityId;
			std::string name;
			std::vector<std::type_index> componentTypes;
			std::vector<Component*> components;

		public:
			Entity();
			Entity(unsigned int id);
			~Entity();

			template<typename T>
			void addComponent(T* component)
			{
				//printf("this: 0x%p\n", this);
				component->entity = this;

				componentTypes.push_back(typeid(T));
				components.push_back(component);

			}

			template<typename T>
			T* getComponent()
			{
				for (unsigned int i = 0; i < componentTypes.size(); i++){
					if (componentTypes[i] == typeid(T)){
						return static_cast<T*>(components[i]);
					}
				}

				return NULL;
			}

			template<typename T>
			std::vector<T*> getComponents()
			{
				std::vector<T*> transforms;
				for (unsigned int i = 0; i < componentTypes.size(); i++){
					if (componentTypes[i] == typeid(T)){
						transforms.push_back(static_cast<T*>(components[i]));
					}
				}

				return transforms;
			}
	};
}

#endif