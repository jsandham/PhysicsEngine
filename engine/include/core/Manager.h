#ifndef __MANAGER_H__
#define __MANAGER_H__

#include <map>
#include <string>

#include "Mesh.h"
#include "GMesh.h"
#include "Material.h"
#include "Shader.h"
#include "Texture2D.h"

#include "../core/Entity.h"
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
#include "../components/Joint.h"
#include "../components/SpringJoint.h"

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

		unsigned int sizeOfEntity;
		unsigned int sizeOfTransform;
		unsigned int sizeOfRigidbody;
		unsigned int sizeOfCamera;
		unsigned int sizeOfMeshRenderer;
		unsigned int sizeOfDirectionalLight;
		unsigned int sizeOfSpotLight;
		unsigned int sizeOfPointLight;
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

		int maxAllowedMaterials;
		int maxAllowedTextures;
		int maxAllowedShaders;
		int maxAllowedMeshes;
		int maxAllowedGMeshes;
	};
#pragma pack(pop)

	struct Scene
	{
		std::string name;
		std::string filepath;
		bool isLoaded;
	};

	struct Asset
	{
		std::string filepath;
	};

	class Manager
	{
		private:
			// entities and components
			int numberOfEntities;
			int numberOfTransforms;
			int numberOfRigidbodies;
			int numberOfCameras;
			int numberOfMeshRenderers;
			int numberOfDirectionalLights;
			int numberOfSpotLights;
			int numberOfPointLights;

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

			Material* getMaterial(int id);
			Shader* getShader(int id);
			Texture2D* getTexture2D(int id);
			Mesh* getMesh(int id);
			GMesh* getGMesh(int id);

			template<typename T>
			T* getComponent(int entityId)
			{
				Entity* entity = getEntity(entityId);

				if(entity == NULL){
					return NULL;
				}

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

						if(componentType == getType<T>()){
							std::map<int, int>::iterator it2 = idToGlobalIndexMap.find(componentId);
							if(it2 != idToGlobalIndexMap.end()){
								componentGlobalIndex = it2->second;
							}
							else{
								std::cout << "Error: When searching entity with id " << entityId << " no component with id " << componentId << " was found in map" << std::endl;
								return NULL;
							}

							// TODO: replace with map
							if(componentType == (int)ComponentType::TransformType){
								return &transforms[componentGlobalIndex];
							}
							else if(componentType == (int)ComponentType::RigidbodyType){
								return &rigidbodies[componentGlobalIndex];
							}
							else if(componentType == (int)ComponentType::CameraType){
								return &cameras[componentGlobalIndex];
							}
							else if(componentType == (int)ComponentType::MeshRendererType){
								return &directionalLights[componentGlobalIndex];
							}
							else if(componentType == (int)ComponentType::DirectionalLightType){
								return &pointLights[componentGlobalIndex];
							}
							else if(componentType == (int)ComponentType::SpotLightType){
								return &spotLights[componentGlobalIndex];
							}
							else if(componentType == (int)ComponentType::PointLightType){
								return &meshRenderers[componentGlobalIndex];
							}
							else{
								return NULL;
							}
						}
					}
				}

				return NULL;
			}

			template<typename T>
			int getType()
			{
				int type = -1;
				if(typeid(T) == typeid(Transform)){
					type = 0;
				}
				else if(typeid(T) == typeid(Rigidbody)){
					type = 1;
				}
				else if(typeid(T) == typeid(Camera)){
					type = 2;
				}
				else if(typeid(T) == typeid(DirectionalLight)){
					type = 3;
				}
				else if(typeid(T) == typeid(PointLight)){
					type = 4;
				}
				else if(typeid(T) == typeid(SpotLight)){
					type = 5;
				}
				else if(typeid(T) == typeid(MeshRenderer)){
					type = 6;
				}
				else if(typeid(T) == typeid(Cloth)){
					type = 7;
				}
				else if(typeid(T) == typeid(Solid)){
					type = 8;
				}
				else if(typeid(T) == typeid(Fluid)){
					type = 9;
				}
				
				return type;
			}


			template<typename T>
			void instantiate()
			{

			}

			void destroy()
			{

			}

		// private:
		// 	template<typename T>
		// 	void setGlobalIndexOnComponent(T* components, int numberOfComponents)
		// 	{
		// 		for(int i = 0; i < numberOfComponents; i++){
		// 			components[i].globalComponentIndex = i;

		// 			int entityId = components[i].entityId;
		// 			int globalEntityIndex = idToGlobalIndexMap.find(entityId)->second;
		// 			components[i].globalEntityIndex = globalEntityIndex;
		// 		}
		// 	}
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