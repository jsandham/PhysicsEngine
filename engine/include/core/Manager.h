#ifndef __MANAGER_H__
#define __MANAGER_H__

#include <map>
#include <string>

#include "Scene.h"
#include "Asset.h"
#include "Entity.h"
#include "Mesh.h"
#include "GMesh.h"
#include "Material.h"
#include "Shader.h"
#include "Texture2D.h"

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
		unsigned int meshId;
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
		unsigned int gmeshId;
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
		int maxAllowedEntities;
		int maxAllowedTransforms;
		int maxAllowedRigidbodies;
		int maxAllowedCameras;
		int maxAllowedMeshRenderers;
		int maxAllowedLineRenderers;
		int maxAllowedDirectionalLights;
		int maxAllowedSpotLights;
		int maxAllowedPointLights;
		int maxAllowedBoxColliders;
		int maxAllowedSphereColliders;
		int maxAllowedCapsuleColliders;

		int maxAllowedMaterials;
		int maxAllowedTextures;
		int maxAllowedShaders;
		int maxAllowedMeshes;
		int maxAllowedGMeshes;
	};
#pragma pack(pop)

	class Manager
	{
		private:
			// entities, components, & systems
			int numberOfEntities;
			int numberOfTransforms;
			int numberOfRigidbodies;
			int numberOfCameras;
			int numberOfMeshRenderers;
			int numberOfLineRenderers;
			int numberOfDirectionalLights;
			int numberOfSpotLights;
			int numberOfPointLights;
			int numberOfBoxColliders;
			int numberOfSphereColliders;
			int numberOfCapsuleColliders;
			int numberOfSystems;

			// number of assets
			int numberOfMaterials;
			int numberOfTextures;
			int numberOfShaders;
			int numberOfMeshes;
			int numberOfGMeshes;

			std::map<int, std::string> assetIdToFilePathMap;
			std::map<int, int> assetIdToGlobalIndexMap;
			std::map<int, int> idToGlobalIndexMap;
			std::map<int, int> componentIdToTypeMap;
			std::map<int, Component*> componentIdToMemoryMap; // to MemoryPointer?
			std::map<int, std::vector<int>> entityIdToComponentIds; 

			BuildSettings settings;

			// entities and components
			Entity* entities;
			Transform* transforms;
			Rigidbody* rigidbodies;
			Camera* cameras;
			MeshRenderer* meshRenderers;
			LineRenderer* lineRenderers;
			DirectionalLight* directionalLights;
			SpotLight* spotLights;
			PointLight* pointLights;
			BoxCollider* boxColliders;
			SphereCollider* sphereColliders;
			CapsuleCollider* capsuleColliders;

			// systems
			std::vector<System*> systems;

			// entities marked for cleanup
			std::vector<int> entitiesMarkedForLatentDestroy;

			// assets
			Material* materials;
			Shader* shaders;
			Texture2D* textures;
			Mesh* meshes;
			GMesh* gmeshes;


		public:
			Manager();
			~Manager();

			bool validate(std::vector<Scene> scenes, std::vector<Asset> assets);
			void load(Scene scene, std::vector<Asset> assets);

			int getNumberOfEntities();
			int getNumberOfTransforms();
			int getNumberOfRigidbodies();
			int getNumberOfCameras();
			int getNumberOfMeshRenderers();
			int getNumberOfLineRenderers();
			int getNumberOfDirectionalLights();
			int getNumberOfSpotLights();
			int getNumberOfPointLights();
			int getNumberOfBoxColliders();
			int getNumberOfSphereColliders();
			int getNumberOfCapsuleColliders();

			int getNumberOfSystems();

			int getNumberOfMaterials();
			int getNumberOfShaders();
			int getNumberOfTextures();
			int getNumberOfMeshes();
			int getNumberOfGmeshes();

			Entity* getEntity(int id);
			Transform* getTransform(int id);
			Rigidbody* getRigidbody(int id);
			Camera* getCamera(int id);
			MeshRenderer* getMeshRenderer(int id);
			LineRenderer* getLineRenderer(int id);
			DirectionalLight* getDirectionalLight(int id);
			SpotLight* getSpotLight(int id);
			PointLight* getPointLight(int id);
			BoxCollider* getBoxCollider(int id);
			SphereCollider* getSphereCollider(int id);
			CapsuleCollider* getCapsuleCollider(int id);

			System* getSystem(int id);

			Material* getMaterial(int id);
			Shader* getShader(int id);
			Texture2D* getTexture2D(int id);
			Mesh* getMesh(int id);
			GMesh* getGMesh(int id);

			Entity* getEntityByIndex(int index);
			Transform* getTransformByIndex(int index);
			Rigidbody* getRigidbodyByIndex(int index);
			Camera* getCameraByIndex(int index);
			MeshRenderer* getMeshRendererByIndex(int index);
			LineRenderer* getLineRendererByIndex(int index);
			DirectionalLight* getDirectionalLightByIndex(int index);
			SpotLight* getSpotLightByIndex(int index);
			PointLight* getPointLightByIndex(int index);
			BoxCollider* getBoxColliderByIndex(int index);
			SphereCollider* getSphereColliderByIndex(int index);
			CapsuleCollider* getCapsuleColliderByIndex(int index);

			System* getSystemByIndex(int index);

			Material* getMaterialByIndex(int index);
			Shader* getShaderByIndex(int index);
			Texture2D* getTexture2DByIndex(int index);
			Mesh* getMeshByIndex(int index);
			GMesh* getGMeshByIndex(int index);

			void latentDestroy(int entityId);
			void immediateDestroy(int entityId);
			bool isMarkedForLatentDestroy(int entityId);
			std::vector<int> getEntitiesMarkedForLatentDestroy();
			Entity* instantiate();
			Entity* instantiate(int entityId);

			template<typename T>
			T* addComponent(int entityId)
			{
				Entity* entity = getEntity(entityId);

				if(entity == NULL){ return NULL; }

				return NULL;
			}

			template<typename T>
			T* getComponent(int entityId)
			{
				Entity* entity = getEntity(entityId);

				if(entity == NULL){ return NULL; }

				std::vector<int> componentsOnEntity;
				std::map<int, std::vector<int>>::iterator it1 = entityIdToComponentIds.find(entityId);
				if(it1 != entityIdToComponentIds.end()){
					componentsOnEntity = it1->second;
				}
				else{
					std::cout << "Error: When searching entity with id " << entityId << " no components were found in entity id to component ids map" << std::endl;
					return NULL;
				}



				for(unsigned int i = 0; i < componentsOnEntity.size(); i++){
					int componentId = componentsOnEntity[i];
					int componentType = -1;
					int componentGlobalIndex = -1;
					if(componentId != -1){
						std::map<int, int>::iterator it2 = componentIdToTypeMap.find(componentId);
						if(it2 != componentIdToTypeMap.end()){
							componentType = it2->second;
						}
						else{
							std::cout << "Error: When searching entity with id " << entityId << " no component with id " << componentId << " was found in component type map" << std::endl;
							return NULL;
						}

						if(componentType == -1){
							std::cout << "Error: When searching entity with id " << entityId << " the component type found corresponding to component " << componentId << " was invalid" << std::endl;
							return NULL;
						}

						if(componentType == Component::getInstanceType<T>()){
							std::map<int, int>::iterator it3 = idToGlobalIndexMap.find(componentId);
							if(it3 != idToGlobalIndexMap.end()){
								componentGlobalIndex = it3->second;
							}
							else{
								std::cout << "Error: When searching entity with id " << entityId << " no component with id " << componentId << " was found in map" << std::endl;
								return NULL;
							}

							std::map<int, Component*>::iterator it4 = componentIdToMemoryMap.find(componentType);
							if(it4 != componentIdToMemoryMap.end()){
								T* component = static_cast<T*>(it4->second) + componentGlobalIndex;
								return component;
							}
							else{
								std::cout << "Error: When searching entity with id: " << entityId << " the component type searched for does not exist in map" << std::endl;
								return NULL;
							}
						}
					}
				}

				return NULL;
			}

	};
}

#endif