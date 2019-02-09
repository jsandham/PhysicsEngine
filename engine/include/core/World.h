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
#include "Pool.h"
#include "Octtree.h"

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
		unsigned int maxAllowedEntities;
		unsigned int maxAllowedTransforms;
		unsigned int maxAllowedRigidbodies;
		unsigned int maxAllowedCameras;
		unsigned int maxAllowedMeshRenderers;
		unsigned int maxAllowedLineRenderers;
		unsigned int maxAllowedDirectionalLights;
		unsigned int maxAllowedSpotLights;
		unsigned int maxAllowedPointLights;
		unsigned int maxAllowedBoxColliders;
		unsigned int maxAllowedSphereColliders;
		unsigned int maxAllowedCapsuleColliders;

		unsigned int maxAllowedMaterials;
		unsigned int maxAllowedTextures;
		unsigned int maxAllowedShaders;
		unsigned int maxAllowedMeshes;
		unsigned int maxAllowedGMeshes;

		int physicsDepth;
		float centre[3];
		float extent[3];
	};
#pragma pack(pop)

	class World
	{
		private:
			// BuildSettings settings; 

			// Pool<Entity>* entities;
			// Pool<Transform>* transforms;
			// Pool<Rigidbody>* rigidbodies;
			// Pool<Camera>* cameras;
			// Pool<MeshRenderer>* meshRenderers;
			// Pool<LineRenderer>* lineRenderers;
			// Pool<DirectionalLight>* directionalLights;
			// Pool<SpotLight>* spotLights;
			// Pool<PointLight>* pointLights;
			// Pool<BoxCollider>* boxColliders;
			// Pool<SphereCollider>* sphereColliders;
			// Pool<CapsuleCollider>* capsuleColliders;

			// Pool<Material>* materials;
			// Pool<Texture2D>* textures;
			// Pool<Shader>* shaders;
			// Pool<Mesh>* meshes;
			// Pool<GMesh>* gmeshes;
			
			Line* line;

			std::vector<System*> systems;

			Bounds* bounds;
			Octtree* physics;

			//std::map<Guid, std::string> assetIdToFilePath;
			std::map<Guid, int> assetIdToGlobalIndex;
			std::map<Guid, int> idToGlobalIndex;
			std::map<Guid, int> componentIdToType;
			std::map<Guid, std::vector<Guid>> entityIdToComponentIds; 
			//std::map<int, void*> componentTypeToPool;
			//std::map<int, void*> assetTypeToPool;

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
				// int componentType = Component::getInstanceType<T>();

				// std::map<int, void*>::iterator it = componentTypeToPool.find(componentType);
				// if(it != componentTypeToPool.end()){
				// 	Pool<T>* pool = static_cast<Pool<T>*>(it->second);

				// 	return pool->getIndex();
				// }
				// else{
				// 	return 0;
				// }
				return 0;
			}

			template<typename T>
			int getNumberOfAssets()
			{
				// int assetType = Asset::getInstanceType<T>();
				// std::map<int, void*>::iterator it1 = assetTypeToPool.find(assetType);
				// if(it1 != assetTypeToPool.end()){
				// 	Pool<T>* pool = static_cast<Pool<T>*>(it1->second);

				// 	return pool->getIndex();
				// }
				// else{
				// 	return 0;
				// }
				return 0;
			}

			Entity* getEntity(Guid id);
			System* getSystem(Guid id);

			template<typename T>
			T* getComponent(Guid entityId)
			{
				// Entity* entity = getEntity(entityId);

				// if(entity == NULL){ return NULL; }

				// std::vector<Guid> componentsOnEntity;
				// std::map<Guid, std::vector<Guid>>::iterator it1 = entityIdToComponentIds.find(entityId);
				// if(it1 != entityIdToComponentIds.end()){
				// 	componentsOnEntity = it1->second;
				// }
				// else{
				// 	std::cout << "Error: When searching entity with id " << entityId.toString() << " no components were found in entity id to component ids map" << std::endl;
				// 	return NULL;
				// }

				// for(unsigned int i = 0; i < componentsOnEntity.size(); i++){
				// 	Guid componentId = componentsOnEntity[i];
				// 	int componentType = -1;
				// 	int componentGlobalIndex = -1;
				// 	if(componentId != Guid::INVALID){
				// 		std::map<Guid, int>::iterator it2 = componentIdToType.find(componentId);
				// 		if(it2 != componentIdToType.end()){
				// 			componentType = it2->second;
				// 		}
				// 		else{
				// 			std::cout << "Error: When searching entity with id " << entityId.toString() << " no component with id " << componentId.toString() << " was found in component type map" << std::endl;//
				// 			return NULL;
				// 		}

				// 		if(componentType == -1){
				// 			std::cout << "Error: When searching entity with id " << entityId.toString() << " the component type found corresponding to component " << componentId.toString() << " was invalid" << std::endl;
				// 			return NULL;
				// 		}

				// 		if(componentType == Component::getInstanceType<T>()){
				// 			std::map<Guid, int>::iterator it3 = idToGlobalIndex.find(componentId);
				// 			if(it3 != idToGlobalIndex.end()){
				// 				componentGlobalIndex = it3->second;
				// 			}
				// 			else{
				// 				std::cout << "Error: When searching entity with id " << entityId.toString() << " no component with id " << componentId.toString() << " was found in map" << std::endl;
				// 				return NULL;
				// 			}

				// 			std::map<int, void*>::iterator it4 = componentTypeToPool.find(componentType);
				// 			if(it4 != componentTypeToPool.end()){
				// 				//PoolAllocator pool = getAllocator<T>();!?!?!!?!? does this work!?!?!?!?
				// 				//return pool.get(componentGlobalIndex);
				// 				Pool<T>* pool = static_cast<Pool<T>*>(it4->second);

				// 				return pool->get(componentGlobalIndex);
				// 			}
				// 			else{
				// 				std::cout << "Error: When searching entity with id: " << entityId.toString() << " the component type searched for does not exist in map" << std::endl;
				// 				return NULL;
				// 			}
				// 		}
				// 	}
				// }

				return NULL;
			}

			template<typename T>
			T* addComponent(Guid entityId)
			{
				// Entity* entity = getEntity(entityId);

				// if(entity == NULL){ return NULL; }

				// int componentGlobalIndex = -1;
				// Guid componentId = Guid::newGuid();
				// T* component = NULL;

				// int componentType = Component::getInstanceType<T>();

				// componentIdToType[componentId] = componentType;

				// std::map<int, void*>::iterator it = componentTypeToPool.find(componentType);
				// if(it != componentTypeToPool.end()){
				// 	Pool<T>* pool = static_cast<Pool<T>*>(it->second);
					
				// 	componentGlobalIndex = pool->getIndex();
				// 	idToGlobalIndex[componentId] = componentGlobalIndex;

				// 	pool->increment();

				// 	component = pool->get(componentGlobalIndex);

				// 	component->entityId = entityId;
				// 	component->componentId = componentId;

				// 	std::map<Guid, std::vector<Guid>>::iterator it2 = entityIdToComponentIds.find(entityId);
				// 	if(it2 != entityIdToComponentIds.end()){
				// 		it2->second.push_back(componentId); 
				// 	}

				// 	component->setManager(this);
				// }
				// else
				// {
				// 	return NULL;
				// }

				// return component;
				return NULL;
			}

			template<typename T>
			T* getAsset(Guid id)
			{
				// int assetType = Asset::getInstanceType<T>();
				// std::map<int, void*>::iterator it1 = assetTypeToPool.find(assetType);
				// if(it1 != assetTypeToPool.end()){
				// 	Pool<T>* pool = static_cast<Pool<T>*>(it1->second);

				// 	std::map<Guid, int>::iterator it2 = assetIdToGlobalIndex.find(id);
				// 	if(it2 != assetIdToGlobalIndex.end()){
				// 		return pool->get(it2->second);
				// 	} 
				// 	else{
				// 		return NULL;
				// 	}
				// }
				// else{
				// 	return NULL;
				// }
				return NULL;
			}

			Entity* getEntityByIndex(int index);
			System* getSystemByIndex(int index);

			template<typename T>
			T* getComponentByIndex(int index)
			{
				// int componentType = Component::getInstanceType<T>();

				// std::map<int, void*>::iterator it = componentTypeToPool.find(componentType);
				// if(it != componentTypeToPool.end()){
				// 	Pool<T>* pool = static_cast<Pool<T>*>(it->second);

				// 	return pool->get(index);
				// }
				// else{
				// 	return NULL;
				// }
				return NULL;
			}

			template<typename T>
			T* getAssetByIndex(int index)
			{
				// int assetType = Asset::getInstanceType<T>();
				// std::map<int, void*>::iterator it1 = assetTypeToPool.find(assetType);
				// if(it1 != assetTypeToPool.end()){
				// 	Pool<T>* pool = static_cast<Pool<T>*>(it1->second);

				// 	return pool->get(index);
				// }
				// else{
				// 	return NULL;
				// }
				return NULL;
			}

			template<typename T>
			T* create() //createAsset??
			{
				// T* asset = NULL;

				// int assetType = Asset::getInstanceType<T>();

				// std::cout << "asset type: " << assetType << std::endl;

				// std::map<int, void*>::iterator it = assetTypeToPool.find(assetType);
				// if(it != assetTypeToPool.end()){
				// 	Pool<T>* pool = static_cast<Pool<T>*>(it->second);

				// 	int index = pool->getIndex();

				// 	pool->increment();

				// 	std::cout << "index: " << index << std::endl;

				// 	Guid assetId = Guid::newGuid();

				// 	assetIdToGlobalIndex[assetId] = index;

				// 	asset = pool->get(index);
				// 	asset->assetId = assetId;

				// 	asset->setManager(this);
				// }
				// else{
				// 	std::cout << "Error: Asset pool does not exist" << std::endl;
				// 	return NULL;
				// }

				// return asset;
				return NULL;
			}

			Line* getLine();
			Bounds* getWorldBounds();
			Octtree* getPhysicsTree();

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