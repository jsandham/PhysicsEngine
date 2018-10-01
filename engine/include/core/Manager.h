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
		size_t sizeOfDirectionalLight;
		size_t sizeOfSpotLight;
		size_t sizeOfPointLight;
		size_t sizeOfBoxCollider;
		size_t sizeOfSphereCollider;
		size_t sizeOfCapsuleCollider;

		//size_t sizeOfAllSystems;
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
		int maxAllowedDirectionalLights;
		int maxAllowedSpotLights;
		int maxAllowedPointLights;
		int maxAllowedBoxColliders;
		int maxAllowedSphereColliders;
		int maxAllowedCapsuleColliders;
		int maxAllowedSystems;

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
			std::map<int, Component*> componentIdToMemoryMap;

			BuildSettings settings;

			// entities and components
			Entity* entities;
			Transform* transforms;
			Rigidbody* rigidbodies;
			Camera* cameras;
			MeshRenderer* meshRenderers;
			DirectionalLight* directionalLights;
			SpotLight* spotLights;
			PointLight* pointLights;
			BoxCollider* boxColliders;
			SphereCollider* sphereColliders;
			CapsuleCollider* capsuleColliders;

			// systems
			std::vector<System*> systems;

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

			template<typename T>
			T* getComponent(int entityId)
			{
				Entity* entity = getEntity(entityId);

				if(entity == NULL){ return NULL; }

				for(int i = 0; i < 8; i++){
					int componentId = entity->componentIds[i];
					int componentType = -1;
					int componentGlobalIndex = -1;
					if(componentId != -1){
						std::map<int, int>::iterator it1 = componentIdToTypeMap.find(componentId);
						if(it1 != componentIdToTypeMap.end()){
							componentType = it1->second;
						}
						else{
							std::cout << "Error: When searching entity with id " << entityId << " no component with id " << componentId << " was found in component type map" << std::endl;
							return NULL;
						}

						if(componentType == -1){
							std::cout << "Error: When searching entity with id " << entityId << " the component type found corresponding to component " << componentId << " was invalid" << std::endl;
							return NULL;
						}

						if(componentType == Component::getType<T>()){
							std::map<int, int>::iterator it2 = idToGlobalIndexMap.find(componentId);
							if(it2 != idToGlobalIndexMap.end()){
								componentGlobalIndex = it2->second;
							}
							else{
								std::cout << "Error: When searching entity with id " << entityId << " no component with id " << componentId << " was found in map" << std::endl;
								return NULL;
							}

							std::map<int, Component*>::iterator it3 = componentIdToMemoryMap.find(componentType);
							if(it3 != componentIdToMemoryMap.end()){
								T* component = static_cast<T*>(it3->second) + componentGlobalIndex;
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


			template<typename T>
			void instantiate()
			{

			}

			void destroy()
			{

			}
	};
}

#endif









































// #ifndef __MANAGER_H__
// #define __MANAGER_H__

// #include <vector>
// #include <string>

// #include "../entities/Entity.h"
// #include "../components/Transform.h"
// #include "../components/Rigidbody.h"
// #include "../components/DirectionalLight.h"
// #include "../components/PointLight.h"
// #include "../components/SpotLight.h"
// #include "../components/MeshRenderer.h"
// #include "../components/Collider.h"
// #include "../components/SphereCollider.h"
// #include "../components/BoxCollider.h"
// #include "../components/Joint.h"
// #include "../components/SpringJoint.h"

// #include "../components/Camera.h"
// #include "../components/Fluid.h"
// #include "../components/Cloth.h"
// #include "../components/Solid.h"

// #include "Pool.h"

// #include "../core/GMesh.h"
// #include "../core/Mesh.h"

// #include "../graphics/Texture2D.h"
// #include "../graphics/Cubemap.h"
// #include "../graphics/Shader.h"
// #include "../graphics/Material.h"

// namespace PhysicsEngine
// {
// 	class Manager
// 	{
// 		private:
// 			// assets
// 			std::vector<GMesh> gmeshes;
// 			std::vector<Mesh> meshes;
// 			std::vector<Texture2D> textures;
// 			std::vector<Cubemap> cubemaps;
// 			std::vector<Shader> shaders;
// 			std::vector<Material> materials;

// 			std::map<std::string, int> gmeshMap;
// 			std::map<std::string, int> meshMap;
// 			std::map<std::string, int> textureMap;
// 			std::map<std::string, int> cubemapMap;
// 			std::map<std::string, int> shaderMap;
// 			std::map<std::string, int> materialMap;

// 			std::vector<Entity*> entities;
// 			std::vector<Transform*> transforms;
// 			std::vector<Rigidbody*> rigidbodies;
// 			std::vector<DirectionalLight*> directionalLights;
// 			std::vector<PointLight*> pointLights;
// 			std::vector<SpotLight*> spotLights;
// 			std::vector<MeshRenderer*> meshRenderers;
// 			// std::vector<LineRenderer*> lineRenderers;
// 			std::vector<Collider*> colliders;
// 			std::vector<SphereCollider*> sphereColliders;
// 			std::vector<BoxCollider*> boxColliders;
// 			std::vector<Joint*> joints;
// 			std::vector<SpringJoint*> springJoints;
// 			std::vector<Fluid*> fluids;
// 			std::vector<Cloth*> cloths;
// 			std::vector<Solid*> solids;

// 			Camera* camera;

// 			Pool<Entity> entityPool;
// 			Pool<Transform> transformPool;
// 			Pool<Rigidbody> rigidbodyPool;
// 			Pool<DirectionalLight> directionalLightPool;
// 			Pool<PointLight> pointLightPool;
// 			Pool<SpotLight> spotLightPool;
// 			Pool<MeshRenderer> meshRendererPool;
// 			//Pool<LineRenderer> lineRendererPool;
// 			Pool<SphereCollider> sphereColliderPool;
// 			Pool<BoxCollider> boxColliderPool;
// 			Pool<SpringJoint> springJointPool;
// 			Pool<Fluid> fluidPool;
// 			Pool<Cloth> clothPool;
// 			Pool<Solid> solidPool;

// 			Pool<Camera> cameraPool;

// 		public:
// 			Manager();
// 			~Manager();

// 			Entity* createEntity();
// 			Transform* createTransform();
// 			Rigidbody* createRigidbody();
// 			DirectionalLight* createDirectionalLight();
// 			PointLight* createPointLight();
// 			SpotLight* createSpotLight();
// 			MeshRenderer* createMeshRenderer();
// 			//LineRenderer* createLineRenderer();
// 			SphereCollider* createSphereCollider();
// 			BoxCollider* createBoxCollider();
// 			SpringJoint* createSpringJoint();
// 			Fluid* createFluid();
// 			Cloth* createCloth();
// 			Solid* createSolid();
// 			Camera* createCamera();

// 			std::vector<Entity*> getEntities();
// 			std::vector<Transform*> getTransforms();
// 			std::vector<Rigidbody*> getRigidbodies();
// 			std::vector<DirectionalLight*> getDirectionalLights();
// 			std::vector<PointLight*> getPointLights();
// 			std::vector<SpotLight*> getSpotLights();
// 			std::vector<MeshRenderer*> getMeshRenderers();
// 			//std::vector<LineRenderer*> getLineRenderers();
// 			std::vector<Collider*> getColliders();
// 			std::vector<SphereCollider*> getSphereColliders();
// 			std::vector<BoxCollider*> getBoxColliders();
// 			std::vector<Joint*> getJoints();
// 			std::vector<SpringJoint*> getSpringJoints();
// 			std::vector<Fluid*> getFluids();
// 			std::vector<Cloth*> getCloths();
// 			std::vector<Solid*> getSolids();
// 			Camera* getCamera();

// 			void loadGMesh(const std::string& name);
// 			void loadMesh(const std::string& name);
// 			void loadTexture2D(const std::string& name);
// 			void loadCubemap(const std::vector<std::string>& names);
// 			void loadShader(const std::string& name, std::string vertex, std::string fragment, std::string geometry = std::string());
// 			void loadMaterial(const std::string& name, Material mat);

// 			GMesh* getGMesh(const std::string& name);
// 			Mesh* getMesh(const std::string& name);
// 			Texture2D* getTexture2D(const std::string& name);
// 			Cubemap* getCubemap(const std::string& name);
// 			Shader* getShader(const std::string& name);
// 			Material* getMaterial(const std::string& name);

// 			GMesh* getGMesh(int filter);
// 			Mesh* getMesh(int filter);
// 			Material* getMaterial(int filter);

// 			std::vector<GMesh>& getGMeshes();
// 			std::vector<Mesh>& getMeshes();
// 			std::vector<Texture2D>& getTextures();
// 			std::vector<Cubemap>& getCubemaps();
// 			std::vector<Shader>& getShaders();
// 			std::vector<Material>& getMaterials();

// 			int getGMeshFilter(const std::string& name);
// 			int getMeshFilter(const std::string& name);
// 			int getMaterialFilter(const std::string& name);

// 			template<typename T>
// 			static int getType()
// 			{
// 				int type = -1;
// 				if(typeid(T) == typeid(Transform)){
// 					type = 0;
// 				}
// 				else if(typeid(T) == typeid(Rigidbody)){
// 					type = 1;
// 				}
// 				else if(typeid(T) == typeid(Camera)){
// 					type = 2;
// 				}
// 				else if(typeid(T) == typeid(DirectionalLight)){
// 					type = 3;
// 				}
// 				else if(typeid(T) == typeid(PointLight)){
// 					type = 4;
// 				}
// 				else if(typeid(T) == typeid(SpotLight)){
// 					type = 5;
// 				}
// 				else if(typeid(T) == typeid(MeshRenderer)){
// 					type = 6;
// 				}
// 				else if(typeid(T) == typeid(Cloth)){
// 					type = 7;
// 				}
// 				else if(typeid(T) == typeid(Solid)){
// 					type = 8;
// 				}
// 				else if(typeid(T) == typeid(Fluid)){
// 					type = 9;
// 				}
				
// 				return type;
// 			}
// 	};
// }

// #endif