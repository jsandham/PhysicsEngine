#ifndef __WORLD_H__
#define __WORLD_H__

#include <map>
#include <string>
#include <assert.h>

#include "Allocator.h"
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
#include "LoadInternal.h"
#include "Util.h"

#include "../components/Transform.h"
#include "../components/MeshRenderer.h"
#include "../components/Camera.h"

#include "../systems/System.h"

namespace PhysicsEngine
{
	class World
	{
		private:
			// allocators for entities, components, systems, and assets
			Allocator* mEntityAllocator;
			std::map<int, Allocator*> mComponentAllocatorMap;
			std::map<int, Allocator*> mSystemAllocatorMap;
			std::map<int, Allocator*> mAssetAllocatorMap;

			// all systems in world listed in order they should be updated
			std::vector<System*> mSystems;

			// world entity, components, system, and asset id state
			std::map<Guid, int> mIdToGlobalIndex;  // could instead map Guid -> std::pair(index, type)?
			std::map<Guid, int> mIdToType;

			// entity ids to component ids
			std::map<Guid, std::vector<std::pair<Guid, int>>> mEntityIdToComponentIds; 

			// entity creation/deletion state
			std::vector<Guid> mEntityIdsMarkedCreated;
			std::vector<Guid> mEntityIdsMarkedLatentDestroy;
			std::vector<std::pair<Guid, int>> mEntityIdsMarkedMoved;

			// component create/deletion state
			std::vector<triple<Guid, Guid, int>> mComponentIdsMarkedCreated;
			std::vector<triple<Guid, Guid, int>> mComponentIdsMarkedLatentDestroy;
			std::vector<triple<Guid, int, int>> mComponentIdsMarkedMoved;

			// asset id to filepath
			std::map<Guid, std::string> assetIdToFilepath;

		public:
			World();
			~World();
			World(const World& other) = delete;
			World& operator=(const World& other) = delete;

			bool loadAsset(const std::string filePath);
			bool loadScene(const std::string filePath, bool ignoreSystems = false);
			bool loadSceneFromEditor(const std::string filePath);

			void latentDestroyEntitiesInWorld();

			int getNumberOfEntities() const;
			int getNumberOfSystems() const;

			template<typename T>
			int getNumberOfComponents(const PoolAllocator<T>* allocator) const
			{
				static_assert(IsComponent<T>::value == true, "'T' is not of type Component");

				return allocator != NULL ? (int)allocator->getCount() : 0;
			}

			template<typename T>
			int getNumberOfComponents() const
			{
				static_assert(IsComponent<T>::value == true, "'T' is not of type Component");

				return getNumberOfComponents<T>(getComponentAllocator_Const<T>());
			}

			template<typename T>
			int getNumberOfAssets(const PoolAllocator<T>* allocator) const
			{
				static_assert(IsAsset<T>::value == true, "'T' is not of type Asset");

				return allocator != NULL ? (int)allocator->getCount() : 0;
			}

			template<typename T>
			int getNumberOfAssets() const
			{
				static_assert(IsAsset<T>::value == true, "'T' is not of type Asset");

				return getNumberOfAssets(getAssetAllocator_Const<T>());
			}

			Entity* getEntity(Guid entityId);

			template<typename T>
			T* getSystem()
			{
				static_assert(IsSystem<T>::value == true, "'T' is not of type System");

				PoolAllocator<T>* allocator = getSystemAllocator<T>();
				if (allocator != NULL) {
					return allocator->get(0);
				}

				return NULL;
			}

			template<typename T>
			T* getComponent(Guid entityId)
			{
				static_assert(IsComponent<T>::value == true, "'T' is not of type Component");

				if (entityId == Guid::INVALID) {
					return NULL;
				}

				std::vector<std::pair<Guid, int>> componentsOnEntity;
				std::map<Guid, std::vector<std::pair<Guid, int>>>::iterator it1 = mEntityIdToComponentIds.find(entityId);
				if(it1 != mEntityIdToComponentIds.end()){
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
			
				if (componentId == Guid::INVALID) {
					return NULL;
				}

				PoolAllocator<T>* allocator = getComponentAllocator<T>();
				if (allocator == NULL) {
					return NULL;
				}

				std::map<Guid, int>::iterator it2 = mIdToGlobalIndex.find(componentId);
				if( it2 != mIdToGlobalIndex.end()){
					return allocator->get(it2->second);
				}

				return NULL;
			}

			template<typename T>
			T* addComponent(Guid entityId)
			{
				static_assert(IsComponent<T>::value == true, "'T' is not of type Component");

				if (entityId == Guid::INVALID) {
					return NULL;
				}

				PoolAllocator<T>* allocator = getComponentOrAddAllocator<T>();

				int componentGlobalIndex = (int)allocator->getCount();
				int componentType = ComponentType<T>::type;
				Guid componentId = Guid::newGuid();
				
				T* component = allocator->construct();

				component->mEntityId = entityId;
				component->mComponentId = componentId;

				mIdToGlobalIndex[componentId] = componentGlobalIndex;
				mIdToType[componentId] = componentType;

				mEntityIdToComponentIds[entityId].push_back(std::make_pair(componentId, componentType));

				mComponentIdsMarkedCreated.push_back(make_triple(entityId, componentId, componentType));

				return component;
			}

			template<typename T>
			T* addComponent(std::vector<char> data)
			{
				static_assert(IsComponent<T>::value == true, "'T' is not of type Component");

				PoolAllocator<T>* allocator = getComponentOrAddAllocator<T>();

				int componentGlobalIndex = (int)allocator->getCount();
				int componentType = ComponentType<T>::type;
			
				T* component = allocator->construct(data);

				mIdToGlobalIndex[component->mComponentId] = componentGlobalIndex;
				mIdToType[component->mComponentId] = componentType;

				mEntityIdToComponentIds[component->mEntityId].push_back(std::make_pair(component->mComponentId, componentType));

				mComponentIdsMarkedCreated.push_back(make_triple(component->mEntityId, component->mComponentId, componentType));

				return component;
			}

			template<typename T>
			T* addSystem(int order)
			{
				static_assert(IsSystem<T>::value == true, "'T' is not of type System");

				PoolAllocator<T>* allocator = getSystemOrAddAllocator<T>();

				int systemGlobalIndex = (int)allocator->getCount();
				int systemType = SystemType<T>::type;
				Guid systemId = Guid::newGuid();

				T* system = allocator->construct();

				system->mSystemId = systemId;

				mIdToGlobalIndex[system->mSystemId] = systemGlobalIndex;
				mIdToType[system->mSystemId] = systemType;

				size_t locationToInsert = mSystems.size();
				for(size_t i = 0; i < mSystems.size(); i++){
					if(order < mSystems[i]->getOrder()){
						locationToInsert = i;
						break;
					}
				}

				mSystems.insert(mSystems.begin() + locationToInsert, system);

				return system;
			}

			template<typename T>
			T* getAsset(Guid assetId)
			{
				static_assert(IsAsset<T>::value == true, "'T' is not of type Asset");

				if (assetId == Guid::INVALID || AssetType<T>::type != getTypeOf(assetId)) {
					return NULL;
				}

				PoolAllocator<T>* allocator = getAssetAllocator<T>();
				if (allocator == NULL) {
					return NULL;
				}

				std::map<Guid, int>::iterator it = mIdToGlobalIndex.find(assetId);
				if(it != mIdToGlobalIndex.end()){
					return allocator->get(it->second);
				}
				else{
					return NULL;
				}
			}

			Entity* getEntityByIndex(int index);
			System* getSystemByIndex(int index);

			template<typename T>
			T* getComponentByIndex(const PoolAllocator<T>* allocator, int index)
			{
				static_assert(IsComponent<T>::value == true, "'T' is not of type Component");

				if (allocator == NULL || index < 0) {
					return NULL;
				}

				return allocator->get(index);
			}

			template<typename T>
			T* getComponentByIndex(int index)
			{
				static_assert(IsComponent<T>::value == true, "'T' is not of type Component");

				return getComponentByIndex(getComponentAllocator_Const<T>(), index);
			}

			template<typename T>
			T* getComponentById(Guid componentId)
			{
				static_assert(IsComponent<T>::value == true, "'T' is not of type Component");

				if (componentId == Guid::INVALID || ComponentType<T>::type != getTypeOf(componentId)) {
					return NULL;
				}

				std::map<Guid, int>::iterator it = mIdToGlobalIndex.find(componentId);
				if(it != mIdToGlobalIndex.end()){
					return getComponentByIndex<T>(it->second);
				}

				return NULL;
			}

			template<typename T>
			T* getAssetByIndex(const PoolAllocator<T>* allocator, int index)
			{
				static_assert(IsAsset<T>::value == true, "'T' is not of type Asset");

				if (allocator == NULL || index < 0) {
					return NULL;
				}

				return allocator->get(index);
			}

			template<typename T>
			T* getAssetByIndex(int index)
			{
				static_assert(IsAsset<T>::value == true, "'T' is not of type Asset");

				return getAssetByIndex(getAssetAllocator_Const<T>(), index);
			}

			int getIndexOf(Guid id) const;
			int getTypeOf(Guid id) const;

			template<typename T>
			T* createAsset()
			{
				static_assert(IsAsset<T>::value == true, "'T' is not of type Asset");

				PoolAllocator<T>* allocator = getAssetOrAddAllocator<T>();

				int index = (int)allocator->getCount();
				int type = AssetType<T>::type;
				Guid id = Guid::newGuid();

				mIdToGlobalIndex[id] = index;
				mIdToType[id] = type;

				T* asset = allocator->construct();

				asset->mAssetId = id;

				return asset;
			}

			template<typename T>
			T* createAsset(std::vector<char> data)
			{
				static_assert(IsAsset<T>::value == true, "'T' is not of type Asset");

				PoolAllocator<T>* allocator = getAssetOrAddAllocator<T>();

				int index = (int)allocator->getCount();
				int type = AssetType<T>::type;
				Guid id = Guid::newGuid();

				mIdToGlobalIndex[id] = index;
				mIdToType[id] = type;

				T* asset = allocator->construct(data);

				asset->mAssetId = id;
			}

			Entity* createEntity();
			Entity* createEntity(std::vector<char> data);

			void latentDestroyEntity(Guid entityId);
			void immediateDestroyEntity(Guid entityId);
			void latentDestroyComponent(Guid entityId, Guid componentId, int componentType);
			void immediateDestroyComponent(Guid entityId, Guid componentId, int componentType);
			bool isMarkedForLatentDestroy(Guid id);
			void clearIdsMarkedCreatedOrDestroyed();
			void clearIdsMarkedMoved();

			std::vector<std::pair<Guid, int>> getComponentsOnEntity(Guid entityId);

			std::vector<Guid> getEntityIdsMarkedCreated() const;
			std::vector<Guid> getEntityIdsMarkedLatentDestroy() const;
			std::vector<std::pair<Guid, int>> getEntityIdsMarkedMoved() const;
			std::vector<triple<Guid, Guid, int>> getComponentIdsMarkedCreated() const;
			std::vector<triple<Guid, Guid, int>> getComponentIdsMarkedLatentDestroy() const;
			std::vector<triple<Guid, int, int>> getComponentIdsMarkedMoved() const;

			std::string getAssetFilepath(Guid assetId) const;

			//bool raycast(glm::vec3 origin, glm::vec3 direction, float maxDistance);
			//bool raycast(glm::vec3 origin, glm::vec3 direction, float maxDistance, Collider** collider);

			const PoolAllocator<Entity>* getEntityAllocator_Const() const;

			template<typename T>
			const PoolAllocator<T>* getComponentAllocator_Const() const
			{
				static_assert(IsComponent<T>::value == true, "'T' is not of type Component");

				std::map<int, Allocator*>::const_iterator it = mComponentAllocatorMap.find(ComponentType<T>::type);
				if (it != mComponentAllocatorMap.end()) {
					return static_cast<PoolAllocator<T>*>(it->second);
				}

				return NULL;
			}

			template<typename T>
			const PoolAllocator<T>* getSystemAllocator_Const() const
			{
				static_assert(IsSystem<T>::value == true, "'T' is not of type System");

				std::map<int, Allocator*>::const_iterator it = mSystemAllocatorMap.find(SystemType<T>::type);
				if (it != mSystemAllocatorMap.end()) {
					return static_cast<PoolAllocator<T>*>(it->second);
				}

				return NULL;
			}

			template<typename T>
			const PoolAllocator<T>* getAssetAllocator_Const() const
			{
				static_assert(IsAsset<T>::value == true, "'T' is not of type Asset");

				std::map<int, Allocator*>::const_iterator it = mAssetAllocatorMap.find(AssetType<T>::type);
				if (it != mAssetAllocatorMap.end()) {
					return static_cast<PoolAllocator<T>*>(it->second);
				}

				return NULL;
			}

			private:
				PoolAllocator<Entity>* getEntityAllocator()
				{
					if (mEntityAllocator == NULL) {
						return NULL;
					}

					return static_cast<PoolAllocator<Entity>*>(mEntityAllocator);
				}

				PoolAllocator<Entity>* getEntityOrAddAllocator()
				{
					PoolAllocator<Entity>* allocator = getEntityAllocator();
					if (allocator == NULL) {
						allocator = new PoolAllocator<Entity>();
						mEntityAllocator = allocator;
					}

					return allocator;
				}

				template<typename T>
				PoolAllocator<T>* getComponentAllocator()
				{
					static_assert(IsComponent<T>::value == true, "'T' is not of type Component");

					std::map<int, Allocator*>::iterator it = mComponentAllocatorMap.find(ComponentType<T>::type);
					if (it != mComponentAllocatorMap.end()) {
						return static_cast<PoolAllocator<T>*>(it->second);
					}

					return NULL;
				}

				template<typename T>
				PoolAllocator<T>* getComponentOrAddAllocator()
				{
					static_assert(IsComponent<T>::value == true, "'T' is not of type Component");

					PoolAllocator<T>* allocator = getComponentAllocator<T>();
					if (allocator == NULL) {
						allocator = new PoolAllocator<T>();
						mComponentAllocatorMap[ComponentType<T>::type] = allocator;
					}

					return allocator;
				}

				template<typename T>
				PoolAllocator<T>* getSystemAllocator()
				{
					static_assert(IsSystem<T>::value == true, "'T' is not of type System");

					std::map<int, Allocator*>::iterator it = mSystemAllocatorMap.find(SystemType<T>::type);
					if (it != mSystemAllocatorMap.end()) {
						return static_cast<PoolAllocator<T>*>(it->second);
					}

					return NULL;
				}

				template<typename T>
				PoolAllocator<T>* getSystemOrAddAllocator()
				{
					static_assert(IsSystem<T>::value == true, "'T' is not of type System");

					PoolAllocator<T>* allocator = getSystemAllocator<T>();
					if (allocator == NULL) {
						allocator = new PoolAllocator<T>();
						mSystemAllocatorMap[SystemType<T>::type] = allocator;
					}

					return allocator;
				}

				template<typename T>
				PoolAllocator<T>* getAssetAllocator()
				{
					static_assert(IsAsset<T>::value == true, "'T' is not of type Asset");

					std::map<int, Allocator*>::iterator it = mAssetAllocatorMap.find(AssetType<T>::type);
					if (it != mAssetAllocatorMap.end()) {
						return static_cast<PoolAllocator<T>*>(it->second);
					}

					return NULL;
				}

				template<typename T>
				PoolAllocator<T>* getAssetOrAddAllocator()
				{
					static_assert(IsAsset<T>::value == true, "'T' is not of type Asset");

					PoolAllocator<T>* allocator = getAssetAllocator<T>();
					if (allocator == NULL) {
						allocator = new PoolAllocator<T>();
						mAssetAllocatorMap[AssetType<T>::type] = allocator;
					}

					return allocator;
				}
	};
}

#endif