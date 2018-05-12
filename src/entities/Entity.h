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
		public:
			static int entityCount;

		private:
			int entityId;
			std::string name;
			std::vector<std::type_index> componentTypes;
			std::vector<Component*> components;

		public:
			Entity();
			Entity(unsigned int id);
			~Entity();

			void clear();

			template<typename T>
			void addComponent(T* component)
			{
				printf("this: 0x%p\n", this);
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


			
	
		


		/*private:
			int entityId;
			std::string name;
			std::vector<std::type_index> componentType;
			std::vector<int> componentIndex;

		public:
			static int entityIdCount;
			static std::vector<Entity> entities;

		private:
			Entity();
			~Entity();

		public:
			static void Destroy(Entity *entity);
			static Entity* Instantiate();

			template<typename T>
			void AddComponent()
			{
				int index = ComponentFactory::addComponent(typeid(T), entityId);  

				componentType.push_back(typeid(T));
				componentIndex.push_back(index);
			}

			template<typename T>
			void RemoveComponent()
			{
				for (unsigned int i = 0; i < componentType.size(); i++){
					if (componentType[i] == typeid(T)){
						ComponentFactory::removeComponent(typeid(T), componentIndex[i]);

						std::type_index tempType = componentType[i];
						int tempIndex = componentIndex[i];

						componentType[i] = componentType[componentType.size() - 1];
						componentIndex[i] = componentIndex[componentIndex.size() - 1];

						componentType.pop_back();
						componentIndex.pop_back();
					}
				}
			}

			template<typename T>
			T* GetComponent()
			{
				for (unsigned int i = 0; i < componentType.size(); i++){
					if (componentType[i] == typeid(T)){
						return static_cast<T*>(ComponentFactory::getComponent(typeid(T), componentIndex[i]));
					}
				}

				return NULL;
			}*/
















		/*private:
			static std::map<std::type_index, int> indiciesToComponentMaps;

		public:
			static int entityIdCount;
			static std::map<int, Entity*> entities;
			static std::vector<std::map<int, std::vector<Component*> >> components;

			int entityId;
			std::string name;

		private:
			Entity();
			~Entity();

		public:
			static void Destroy(Entity *entity);
			static Entity* Instantiate();

			template<typename T>
			void AddComponent()
			{
				std::map<std::type_index, int>::iterator it = indiciesToComponentMaps.find(typeid(T));
				if (it != indiciesToComponentMaps.end()){
					int indexToComponentMap = it->second;

					std::map<int, std::vector<Component*>> *cmap = &components[indexToComponentMap];

					(*cmap)[entityId].push_back(ComponentFactory::create(typeid(T), this));
				}
				else{
					std::cout << "Entity: You are trying to add an invalid component to the entity with id " << entityId << std::endl;
				}
			}

			template<typename T>
			void RemoveComponent()
			{
				if (typeid(T) == typeid(Transform)){
					std::cout << "Enity: cannot remove Transform component from entity" << std::endl;
					return;
				}

				std::map<std::type_index, int>::iterator it = indiciesToComponentMaps.find(typeid(T));
				if (it != indiciesToComponentMaps.end()){
					int indexToComponentMap = it->second;

					std::map<int, std::vector<Component*>> *cmap = &components[indexToComponentMap];

					std::map<int, std::vector<Component*>>::iterator it = cmap->find(entityId);
					if (it != cmap->end()){
						std::vector<Component*> *cvec = &(it->second);

						if (cvec->size() == 0){
							cmap->erase(it);

							return;
						}

						Component *c = (*cvec)[cvec->size() - 1];

						delete c;

						cvec->pop_back();

						if (cvec->size() == 0){
							cmap->erase(it);

							return;
						}
					}
				}
				else{
					std::cout << "Entity: you are trying to remove an invalid component from entity with id " << entityId << std::endl;
				}
			}

			template<typename T>
			T* GetComponent()
			{
				std::map<std::type_index, int>::iterator it = indiciesToComponentMaps.find(typeid(T));
				if (it != indiciesToComponentMaps.end()){
					int indexToComponentMap = it->second;

					std::map<int, std::vector<Component*>> *cmap = &components[indexToComponentMap];

					std::map<int, std::vector<Component*>>::iterator it = cmap->find(entityId);
					if (it != cmap->end()){
						std::vector<Component*> *cvec = &(it->second);

						if (cvec->size() > 0){
							return static_cast<T*>((*cvec)[0]);
						}
					}
				}
				else{
					std::cout << "Entity: you are trying to get an invalid component from the entity with id " << entityId << std::endl;
				}

				return NULL;
			}

			template<typename T>
			std::vector<T*> GetComponents()
			{
				std::vector<T*> componentsOfTypeT;

				std::map<std::type_index, int>::iterator it = indiciesToComponentMaps.find(typeid(T));
				if (it != indiciesToComponentMaps.end()){
					int indexToComponentMap = it->second;

					std::map<int, std::vector<Component*>> *cmap = &components[indexToComponentMap];

					std::map<int, std::vector<Component*>>::iterator it = cmap->find(entityId);
					if (it != cmap->end()){
						std::vector<Component*> *cvec = &(it->second);

						for (unsigned int i = 0; i < cvec->size(); i++){
							componentsOfTypeT.push_back(static_cast<T*>((*cvec)[i]));
						}
					}
				}
				else{
					std::cout << "Entity: you are trying to get invalid components from the entity with id " << entityId << std::endl;
				}

				return componentsOfTypeT;
				
			}

		private:
			static std::map<std::type_index, int> createIndiciesToComponentMaps();
			static std::vector<std::map<int, std::vector<Component*> >> createComponentMaps();*/
	};
}

#endif