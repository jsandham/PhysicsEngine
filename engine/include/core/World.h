#ifndef __WORLD_H__
#define __WORLD_H__

#include <map>
#include <string>

#include "PoolAllocator.h"
#include "Scene.h"
#include "Asset.h"
#include "Entity.h"
#include "Mesh.h"
#include "GMesh.h"
#include "Line.h"
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
#include "../components/DirectionalLight.h"
#include "../components/PointLight.h"
#include "../components/SpotLight.h"
#include "../components/MeshRenderer.h"
#include "../components/LineRenderer.h"
#include "../components/Collider.h"
#include "../components/SphereCollider.h"
#include "../components/BoxCollider.h"
#include "../components/CapsuleCollider.h"
#include "../components/Joint.h"
#include "../components/SpringJoint.h"

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

	class World
	{
		private:
			std::vector<System*> systems;

			Bounds bounds;
			Octtree stree; // octtree for static colliders
			Octtree dtree; // octtree for dynamic colliders
			UniformGrid sgrid; // uniform grid for static colliders

			std::map<Guid, int> assetIdToGlobalIndex;
			std::map<Guid, int> idToGlobalIndex;
			std::map<Guid, std::vector<std::pair<Guid, int>>> entityIdToComponentIds; 

			// entities marked for cleanup
			std::vector<Guid> entityIdsMarkedForLatentDestroy;

		public:
			bool debug;

		public:
			World();
			~World();

			bool load(Scene scene, AssetBundle assetBundle);

			int getNumberOfEntities();
			int getNumberOfSystems();

			template<typename T>
			int getNumberOfComponents()
			{
				return (int)getAllocator<T>().getCount();
			}

			template<typename T>
			int getNumberOfAssets()
			{
				return (int)getAllocator<T>().getCount();
			}

			Entity* getEntity(Guid id);
			System* getSystem(Guid id);

			template<typename T>
			T* getComponent(Guid entityId)
			{
				std::vector<std::pair<Guid, int>> componentsOnEntity;
				std::map<Guid, std::vector<std::pair<Guid, int>>>::iterator it1 = entityIdToComponentIds.find(entityId);
				if(it1 != entityIdToComponentIds.end()){
					componentsOnEntity = it1->second;
				}
				else{
					std::cout << "Error: When calling get component on entity with id " << entityId.toString() << " the entity could not be found in the map" << std::endl;
					return NULL;
				}

				Guid componentId = Guid::INVALID;
				for(size_t i = 0; i < componentsOnEntity.size(); i++){
					if(Component::getInstanceType<T>() == componentsOnEntity[i].second){
						componentId = componentsOnEntity[i].first;
						break;
					}	
				}

				if(componentId == Guid::INVALID){
					std::cout << "Error: When calling get component on entity with id " << entityId.toString() << " the component found had an invalid guid id" << std::endl;
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
				int componentGlobalIndex = (int)getAllocator<T>().getCount();
				int componentType = Component::getInstanceType<T>();
				Guid componentId = Guid::newGuid();
				
				std::cout << "componentGlobalIndex: " << componentGlobalIndex << " componentType: " << componentType << " componentId: " << componentId.toString() << std::endl;

				T* component = create<T>();//new T;//static_cast<T*>(getAllocator<T>().allocate());

				component->entityId = entityId;
				component->componentId = componentId;

				idToGlobalIndex[componentId] = componentGlobalIndex;

				entityIdToComponentIds[entityId].push_back(std::make_pair(componentId, componentType));

				return component;
			}

			template<typename T>
			T* getAsset(Guid id)
			{
				std::map<Guid, int>::iterator it = assetIdToGlobalIndex.find(id);
				if(it != assetIdToGlobalIndex.end()){
					return getAllocator<T>().get(it->second);
				}
				else{
					//std::cout << "Error: No asset with id " << id.toString() << " was found" << std::endl;
					return NULL;
				}
			}

			Entity* getEntityByIndex(int index);
			System* getSystemByIndex(int index);

			template<typename T>
			T* getComponentByIndex(int index)
			{
				return getAllocator<T>().get(index);
			}

			template<typename T>
			T* getAssetByIndex(int index)
			{
				return getAllocator<T>().get(index);
			}

			template<typename T>
			T* createAsset()
			{
				int index = (int)getAllocator<T>().getCount();
				Guid id = Guid::newGuid();

				std::map<Guid, int>::iterator it = assetIdToGlobalIndex.find(id);
				if(it == assetIdToGlobalIndex.end()){
					assetIdToGlobalIndex[id] = index;
				}
				else{
					std::cout << "Error: Newly created asset id (" << id.toString() << ") already exists in map?" << std::endl;
					return NULL;
				}

				T* asset = create<T>();//new T;

				asset->assetId = id;

				return asset;
			}

			Bounds* getWorldBounds();
			Octtree* getStaticPhysicsTree();
			Octtree* getDynamicPhysicsTree();
			UniformGrid* getStaticPhysicsGrid();

			bool raycast(glm::vec3 origin, glm::vec3 direction, float maxDistance);
			bool raycast(glm::vec3 origin, glm::vec3 direction, float maxDistance, Collider** collider);


			void latentDestroy(Guid entityId);
			void immediateDestroy(Guid entityId);
			bool isMarkedForLatentDestroy(Guid entityId);
			std::vector<Guid> getEntitiesMarkedForLatentDestroy();
			Entity* instantiate();
			Entity* instantiate(Guid entityId);











			static bool writeToBMP(const std::string& filepath, std::vector<unsigned char>& data, int width, int height, int numChannels);
	};
}

#endif