#ifndef __MANAGER_H__
#define __MANAGER_H__

#include <map>
#include <string>

#include "Scene.h"
#include "Entity.h"
#include "Mesh.h"
#include "GMesh.h"
#include "Line.h"
#include "Material.h"
#include "Shader.h"
#include "Texture2D.h"
#include "Guid.h"
#include "Pool.h"

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
	struct SceneHeader
	{
		unsigned short fileType;
		unsigned int fileSize;

		unsigned int numberOfEntities;
		unsigned int numberOfTransforms;
		unsigned int numberOfRigidbodies;
		unsigned int numberOfCameras;
		unsigned int numberOfMeshRenderers;
		unsigned int numberOfLineRenderers;
		unsigned int numberOfDirectionalLights;
		unsigned int numberOfSpotLights;
		unsigned int numberOfPointLights;
		unsigned int numberOfBoxColliders;
		unsigned int numberOfSphereColliders;
		unsigned int numberOfCapsuleColliders;

		unsigned int numberOfSystems;

		size_t sizeOfEntity;
		size_t sizeOfTransform;
		size_t sizeOfRigidbody;
		size_t sizeOfCamera;
		size_t sizeOfMeshRenderer;
		size_t sizeOfLineRenderer;
		size_t sizeOfDirectionalLight;
		size_t sizeOfSpotLight;
		size_t sizeOfPointLight;
		size_t sizeOfBoxCollider;
		size_t sizeOfSphereCollider;
		size_t sizeOfCapsuleCollider;
	};
#pragma pack(pop)

#pragma pack(push, 1)
	struct MeshHeader
	{
		unsigned short fileType;
		unsigned int fileSize;
		Guid meshId;
		unsigned int verticesSize;
		unsigned int normalsSize;
		unsigned int texCoordsSize;
	};
#pragma pack(pop)

#pragma pack(push, 1)
	struct GMeshHeader
	{
		unsigned short fileType;
		unsigned int fileSize;
		Guid gmeshId;
		int dim;
		int ng;
	    int n;
	    int nte;
	    int ne;
	    int ne_b;
	    int npe;
	    int npe_b;
	    int type;
	    int type_b;
		unsigned int verticesSize;
		unsigned int connectSize;
		unsigned int bconnectSize;
		unsigned int groupsSize;
	};
#pragma pack(pop)

#pragma pack(push, 1)
	struct BuildSettings
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
	};
#pragma pack(pop)

	class Manager
	{
		private:
			BuildSettings settings;

			Pool<Entity>* entities;
			Pool<Transform>* transforms;
			Pool<Rigidbody>* rigidbodies;
			Pool<Camera>* cameras;
			Pool<MeshRenderer>* meshRenderers;
			Pool<LineRenderer>* lineRenderers;
			Pool<DirectionalLight>* directionalLights;
			Pool<SpotLight>* spotLights;
			Pool<PointLight>* pointLights;
			Pool<BoxCollider>* boxColliders;
			Pool<SphereCollider>* sphereColliders;
			Pool<CapsuleCollider>* capsuleColliders;

			Pool<Material>* materials;
			Pool<Texture2D>* textures;
			Pool<Shader>* shaders;
			Pool<Mesh>* meshes;
			Pool<GMesh>* gmeshes;
			
			Line* line;

			std::vector<System*> systems;

			std::map<Guid, std::string> assetIdToFilePath;
			std::map<Guid, int> assetIdToGlobalIndex;
			std::map<Guid, int> idToGlobalIndex;
			std::map<Guid, int> componentIdToType;
			std::map<Guid, std::vector<Guid>> entityIdToComponentIds; 
			std::map<int, void*> componentTypeToPool;

			std::map<int, void*> assetTypeToPool;

			// entities marked for cleanup
			std::vector<Guid> entityIdsMarkedForLatentDestroy;

		public:
			Manager();
			~Manager();

			bool validate(std::vector<Scene> scenes, std::vector<AssetFile> assetFiles);
			void load(Scene scene, std::vector<AssetFile> assetFiles);

			int getNumberOfEntities();
			int getNumberOfSystems();

			template<typename T>
			int getNumberOfComponents()
			{
				int componentType = Component::getInstanceType<T>();

				std::map<int, void*>::iterator it = componentTypeToPool.find(componentType);
				if(it != componentTypeToPool.end()){
					Pool<T>* pool = static_cast<Pool<T>*>(it->second);

					return pool->getIndex();
				}
				else{
					return 0;
				}
			}

			template<typename T>
			int getNumberOfAssets()
			{
				int assetType = Asset::getInstanceType<T>();
				std::map<int, void*>::iterator it1 = assetTypeToPool.find(assetType);
				if(it1 != assetTypeToPool.end()){
					Pool<T>* pool = static_cast<Pool<T>*>(it1->second);

					return pool->getIndex();
				}
				else{
					return 0;
				}
			}

			Entity* getEntity(Guid id);
			System* getSystem(Guid id);

			template<typename T>
			T* getComponent(Guid entityId)
			{
				Entity* entity = getEntity(entityId);

				if(entity == NULL){ return NULL; }

				std::vector<Guid> componentsOnEntity;
				std::map<Guid, std::vector<Guid>>::iterator it1 = entityIdToComponentIds.find(entityId);
				if(it1 != entityIdToComponentIds.end()){
					componentsOnEntity = it1->second;
				}
				else{
					std::cout << "Error: When searching entity with id " << entityId.toString() << " no components were found in entity id to component ids map" << std::endl;
					return NULL;
				}

				for(unsigned int i = 0; i < componentsOnEntity.size(); i++){
					Guid componentId = componentsOnEntity[i];
					int componentType = -1;
					int componentGlobalIndex = -1;
					if(componentId != Guid::INVALID){
						std::map<Guid, int>::iterator it2 = componentIdToType.find(componentId);
						if(it2 != componentIdToType.end()){
							componentType = it2->second;
						}
						else{
							std::cout << "Error: When searching entity with id " << entityId.toString() << " no component with id " << componentId.toString() << " was found in component type map" << std::endl;//
							return NULL;
						}

						if(componentType == -1){
							std::cout << "Error: When searching entity with id " << entityId.toString() << " the component type found corresponding to component " << componentId.toString() << " was invalid" << std::endl;
							return NULL;
						}

						if(componentType == Component::getInstanceType<T>()){
							std::map<Guid, int>::iterator it3 = idToGlobalIndex.find(componentId);
							if(it3 != idToGlobalIndex.end()){
								componentGlobalIndex = it3->second;
							}
							else{
								std::cout << "Error: When searching entity with id " << entityId.toString() << " no component with id " << componentId.toString() << " was found in map" << std::endl;
								return NULL;
							}

							std::map<int, void*>::iterator it4 = componentTypeToPool.find(componentType);
							if(it4 != componentTypeToPool.end()){
								Pool<T>* pool = static_cast<Pool<T>*>(it4->second);

								return pool->get(componentGlobalIndex);
							}
							else{
								std::cout << "Error: When searching entity with id: " << entityId.toString() << " the component type searched for does not exist in map" << std::endl;
								return NULL;
							}
						}
					}
				}

				return NULL;
			}

			template<typename T>
			T* addComponent(Guid entityId)
			{
				Entity* entity = getEntity(entityId);

				if(entity == NULL){ return NULL; }

				int componentGlobalIndex = -1;
				Guid componentId = Guid::newGuid();
				T* component = NULL;

				int componentType = Component::getInstanceType<T>();

				componentIdToType[componentId] = componentType;

				std::map<int, void*>::iterator it = componentTypeToPool.find(componentType);
				if(it != componentTypeToPool.end()){
					Pool<T>* pool = static_cast<Pool<T>*>(it->second);
					
					componentGlobalIndex = pool->getIndex();
					idToGlobalIndex[componentId] = componentGlobalIndex;

					pool->allocate();

					component = pool->get(componentGlobalIndex);

					component->entityId = entityId;
					component->componentId = componentId;

					std::map<Guid, std::vector<Guid>>::iterator it2 = entityIdToComponentIds.find(entityId);
					if(it2 != entityIdToComponentIds.end()){
						it2->second.push_back(componentId); 
					}

					component->setManager(this);
				}
				else
				{
					return NULL;
				}

				return component;
			}

			template<typename T>
			T* getAsset(Guid id)
			{
				int assetType = Asset::getInstanceType<T>();
				std::map<int, void*>::iterator it1 = assetTypeToPool.find(assetType);
				if(it1 != assetTypeToPool.end()){
					Pool<T>* pool = static_cast<Pool<T>*>(it1->second);

					std::map<Guid, int>::iterator it2 = assetIdToGlobalIndex.find(id);
					if(it2 != assetIdToGlobalIndex.end()){
						return pool->get(it2->second);
					} 
					else{
						return NULL;
					}
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
				int componentType = Component::getInstanceType<T>();

				std::map<int, void*>::iterator it = componentTypeToPool.find(componentType);
				if(it != componentTypeToPool.end()){
					Pool<T>* pool = static_cast<Pool<T>*>(it->second);

					return pool->get(index);
				}
				else{
					return NULL;
				}
			}

			template<typename T>
			T* getAssetByIndex(int index)
			{
				int assetType = Asset::getInstanceType<T>();
				std::map<int, void*>::iterator it1 = assetTypeToPool.find(assetType);
				if(it1 != assetTypeToPool.end()){
					Pool<T>* pool = static_cast<Pool<T>*>(it1->second);

					return pool->get(index);
				}
				else{
					return NULL;
				}
			}

			template<typename T>
			T* create() //createAsset??
			{
				T* asset = NULL;

				int assetType = Asset::getInstanceType<T>();

				std::cout << "asset type: " << assetType << std::endl;

				std::map<int, void*>::iterator it = assetTypeToPool.find(assetType);
				if(it != assetTypeToPool.end()){
					Pool<T>* pool = static_cast<Pool<T>*>(it->second);

					int index = pool->getIndex();

					pool->allocate();

					std::cout << "index: " << index << std::endl;

					Guid assetId = Guid::newGuid();

					assetIdToGlobalIndex[assetId] = index;

					asset = pool->get(index);
					asset->assetId = assetId;
				}
				else{
					std::cout << "Error: Asset pool does not exist" << std::endl;
					return NULL;
				}

				return asset;
			}

			Line* getLine();

			void latentDestroy(Guid entityId);
			void immediateDestroy(Guid entityId);
			bool isMarkedForLatentDestroy(Guid entityId);
			std::vector<Guid> getEntitiesMarkedForLatentDestroy();
			Entity* instantiate();
			Entity* instantiate(Guid entityId);
	};
}

#endif