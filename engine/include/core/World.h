#ifndef __WORLD_H__
#define __WORLD_H__

#include <map>
#include <string>
#include <assert.h>

#include "PoolAllocator.h"
#include "Log.h"
#include "Scene.h"
#include "Asset.h"
#include "Entity.h"
#include "Mesh.h"
#include "Material.h"
#include "Shader.h"
#include "Texture2D.h"
#include "Guid.h"
#include "Input.h"
#include "Octtree.h"
#include "UniformGrid.h"
#include "LoadInternal.h"

#include "../components/Transform.h"
#include "../components/Rigidbody.h"
#include "../components/Camera.h"
#include "../components/Light.h"
#include "../components/MeshRenderer.h"
#include "../components/LineRenderer.h"
#include "../components/Collider.h"
#include "../components/SphereCollider.h"
#include "../components/BoxCollider.h"
#include "../components/MeshCollider.h"
#include "../components/CapsuleCollider.h"

#include "../systems/System.h"

namespace PhysicsEngine
{
#pragma pack(push, 1)
	struct BuildSettings // rename to WorldSettings or maybe even get rid of??
	{
		int physicsDepth;
		float centre[3];
		float extent[3];
	};
#pragma pack(pop)

	template<typename T, typename U, typename V>
	struct triple
	{
		T first;
		U second;
		V third;
	};

	template<typename T, typename U, typename V>
	triple<T, U, V> make_triple(T first, U second, V third)
	{
		triple<T, U, V> triple;
		triple.first = first;
		triple.second = second;
		triple.third = third;

		return triple;
	}

	class World
	{
		private:
			//Guid worldId;
			std::vector<System*> systems;

			Bounds bounds;
			//Octtree stree; // octtree for static colliders
			//Octtree dtree; // octtree for dynamic colliders
			UniformGrid sgrid; // uniform grid for static colliders

			// world entity, components, system, and asset id state
			std::map<Guid, int> idToGlobalIndex;
			std::map<Guid, int> idToType;

			// entity ids to component ids map
			std::map<Guid, std::vector<std::pair<Guid, int>>> entityIdToComponentIds; 

			// entity and component creation/deletion state
			std::vector<Guid> entityIdsMarkedCreated;
			std::vector<Guid> entityIdsMarkedLatentDestroy;
			std::vector<std::pair<Guid, int>> entityIdsMarkedMoved;
			std::vector<triple<Guid, Guid, int>> componentIdsMarkedCreated;
			std::vector<triple<Guid, Guid, int>> componentIdsMarkedLatentDestroy;
			std::vector<triple<Guid, int, int>> componentIdsMarkedMoved;

		public:
			bool debug;
			int debugView;

		public:
			World();
			~World();
			World(const World& other) = delete;
			World& operator=(const World& other) = delete;

			bool loadAsset(std::string filePath);
			bool loadScene(std::string filePath, bool ignoreSystems = false);
			bool loadSceneFromEditor(std::string filePath);

			void latentDestroyEntitiesInWorld(); // clear world of entities and components
			//void clearAll(); // clear world of entities, components, systems, and assets

			int getNumberOfEntities();
			int getNumberOfSystems();

			template<typename T>
			int getNumberOfComponents()
			{
				assert(IsComponent<T>::value == true);

				return (int)getAllocator<T>().getCount();
			}

			template<typename T>
			int getNumberOfAssets()
			{
				assert(IsAsset<T>::value == true);

				return (int)getAllocator<T>().getCount();
			}

			Entity* getEntity(Guid id);

			template<typename T>
			T* getFirstSystem()
			{
				return getAllocator<T>().get(0);
			}

			template<typename T>
			T* getSystem(Guid systemId)
			{
				assert(IsSystem<T>::value == true);

				if (systemId == Guid::INVALID) {
					std::string message = "Error: When calling get system with id " + systemId.toString() + " the system found has an invalid guid id\n";
					Log::error(message.c_str());
					return NULL;
				}

				std::map<Guid, int>::iterator it = idToGlobalIndex.find(systemId);
				if (it != idToGlobalIndex.end()) {
					int globalIndex = it->second;

					return getAllocator<T>().get(globalIndex);
				}

				return NULL;
			}

			template<typename T>
			T* getComponent(Guid entityId)
			{
				assert(IsComponent<T>::value == true);

				std::vector<std::pair<Guid, int>> componentsOnEntity;
				std::map<Guid, std::vector<std::pair<Guid, int>>>::iterator it1 = entityIdToComponentIds.find(entityId);
				if(it1 != entityIdToComponentIds.end()){
					componentsOnEntity = it1->second;
				}
				else{
					std::string message = "Error: When calling get component on entity with id " + entityId.toString() + " the entity could not be found in the map\n";
					Log::error(message.c_str());
					return NULL;
				}

				Guid componentId = Guid::INVALID;
				for(size_t i = 0; i < componentsOnEntity.size(); i++){
					if(ComponentType<T>::type == componentsOnEntity[i].second){
						componentId = componentsOnEntity[i].first;
						break;
					}	
				}

				if(componentId == Guid::INVALID){
					std::string message = "Error: When calling get component on entity with id " + entityId.toString() + " the component found had an invalid guid id\n";
					Log::error(message.c_str());
					return NULL;
				}

				std::map<Guid, int>::iterator it2 = idToGlobalIndex.find(componentId);
				if( it2 != idToGlobalIndex.end()){
					int globalIndex = it2->second;

					return getAllocator<T>().get(globalIndex);
				}

				return NULL;
			}

			template<typename T>
			T* addComponent(Guid entityId)
			{
				assert(IsComponent<T>::value == true);

				int componentGlobalIndex = (int)getAllocator<T>().getCount();
				int componentType = ComponentType<T>::type;
				Guid componentId = Guid::newGuid();
				
				T* component = create<T>();//new T;//static_cast<T*>(getAllocator<T>().allocate());

				component->entityId = entityId;
				component->componentId = componentId;

				idToGlobalIndex[componentId] = componentGlobalIndex;
				idToType[componentId] = componentType;

				entityIdToComponentIds[entityId].push_back(std::make_pair(componentId, componentType));

				componentIdsMarkedCreated.push_back(make_triple(entityId, componentId, componentType));

				return component;
			}

			template<typename T>
			T* addComponent(std::vector<char> data)
			{
				assert(IsComponent<T>::value == true);

				int componentGlobalIndex = (int)getAllocator<T>().getCount();
				int componentType = ComponentType<T>::type;
			
				T* component = create<T>(data);

				idToGlobalIndex[component->componentId] = componentGlobalIndex;
				idToType[component->componentId] = componentType;

				entityIdToComponentIds[component->entityId].push_back(std::make_pair(component->componentId, componentType));

				componentIdsMarkedCreated.push_back(make_triple(component->entityId, component->componentId, componentType));

				return component;
			}

			template<typename T>
			T* addSystem(int order)
			{
				assert(IsSystem<T>::value == true);

				int systemGlobalIndex = (int)getAllocator<T>().getCount();
				int systemType = SystemType<T>::type;
				Guid systemId = Guid::newGuid();

				T* system = create<T>();

				system->systemId = systemId;

				idToGlobalIndex[system->systemId] = systemGlobalIndex;
				idToType[system->systemId] = systemType;

				size_t locationToInsert = systems.size();
				for(size_t i = 0; i < systems.size(); i++){
					if(order < systems[i]->getOrder()){
						locationToInsert = i;
						break;
					}
				}

				systems.insert(systems.begin() + locationToInsert, system);

				return system;
			}


			// TODO: Major bug here (and in getComponentById) where the index we find may not correspond to an asset that is on type T. Need something like assetIdToType map?
			template<typename T>
			T* getAsset(Guid id)
			{
				assert(IsAsset<T>::value == true);

				std::map<Guid, int>::iterator it = idToGlobalIndex.find(id);
				if(it != idToGlobalIndex.end()){
					return getAllocator<T>().get(it->second);
				}
				else{
					return NULL;
				}
			}

			Entity* getEntityByIndex(int index);
			System* getSystemByIndex(int index);

			template<typename T>
			T* getComponentByIndex(int index)
			{
				assert(IsComponent<T>::value == true);

				return getAllocator<T>().get(index);
			}

			template<typename T>
			T* getComponentById(Guid componentId)
			{
				assert(IsComponent<T>::value == true);

				std::map<Guid, int>::iterator it = idToGlobalIndex.find(componentId);
				if(it != idToGlobalIndex.end()){
					return getComponentByIndex<T>(it->second);
				}

				return NULL;
			}

			template<typename T>
			T* getAssetByIndex(int index)
			{
				assert(IsAsset<T>::value == true);

				return getAllocator<T>().get(index);
			}

			int getIndexOf(Guid id);
			int getTypeOf(Guid id);

			template<typename T>
			T* createAsset()
			{
				assert(IsAsset<T>::value == true);

				int index = (int)getAllocator<T>().getCount();
				int type = AssetType<T>::type;
				Guid id = Guid::newGuid();

				idToGlobalIndex[id] = index;
				idToType[id] = type;

				T* asset = create<T>();

				asset->assetId = id;

				return asset;
			}

			Entity* createEntity();
			Entity* createEntity(Guid entityId);
			Entity* createEntity(std::vector<char> data);

			Camera* createEditorCamera();

			void latentDestroyEntity(Guid entityId);
			void immediateDestroyEntity(Guid entityId);
			void latentDestroyComponent(Guid entityId, Guid componentId, int componentType);
			void immediateDestroyComponent(Guid entityId, Guid componentId, int componentType);
			bool isMarkedForLatentDestroy(Guid id);
			void clearIdsMarkedCreatedOrDestroyed();
			void clearIdsMarkedMoved();

			std::vector<std::pair<Guid, int>> getComponentsOnEntity(Guid entityId);

			std::vector<Guid> getEntityIdsMarkedCreated();
			std::vector<Guid> getEntityIdsMarkedLatentDestroy();
			std::vector<std::pair<Guid, int>> getEntityIdsMarkedMoved();
			std::vector<triple<Guid, Guid, int>> getComponentIdsMarkedCreated();
			std::vector<triple<Guid, Guid, int>> getComponentIdsMarkedLatentDestroy();
			std::vector<triple<Guid, int, int>> getComponentIdsMarkedMoved();







			Bounds* getWorldBounds();
			//Octtree* getStaticPhysicsTree();
			//Octtree* getDynamicPhysicsTree();
			UniformGrid* getStaticPhysicsGrid();

			bool raycast(glm::vec3 origin, glm::vec3 direction, float maxDistance);
			bool raycast(glm::vec3 origin, glm::vec3 direction, float maxDistance, Collider** collider);
	};
}

#endif