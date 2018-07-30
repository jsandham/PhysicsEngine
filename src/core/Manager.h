#ifndef __MANAGER_H__
#define __MANAGER_H__

#include <map>
#include <string>

#include "SceneSettings.h"

#include "../entities/Entity.h"
#include "../components/Transform.h"
#include "../components/Rigidbody.h"
#include "../components/DirectionalLight.h"
#include "../components/PointLight.h"
#include "../components/SpotLight.h"
#include "../components/MeshRenderer.h"
#include "../components/Collider.h"
#include "../components/SphereCollider.h"
#include "../components/BoxCollider.h"
#include "../components/Joint.h"
#include "../components/SpringJoint.h"
#include "../components/Camera.h"

#include "../core/Material.h"

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
		unsigned int numberOfMeshRenderers;
		unsigned int numberOfDirectionalLights;
		unsigned int numberOfSpotLights;
		unsigned int numberOfPointLights;
		unsigned int sizeOfEntity;
		unsigned int sizeOfTransform;
		unsigned int sizeOfRigidbody;
		unsigned int sizeOfMeshRenderer;
		unsigned int sizeOfDirectionalLight;
		unsigned int sizeOfSpotLight;
		unsigned int sizeOfPointLight;
	};
#pragma pack(pop)

#pragma pack(push, 1)
	struct MaterialHeader
	{
		unsigned short fileType;
		unsigned int fileSize;
		unsigned int materialId;
		unsigned int shaderId;
		unsigned int textureId;
		
	};
#pragma pack(pop)

// #pragma pack(push, 1)
// 	struct ShaderHeader
// 	{
// 		unsigned short fileType;
// 		unsigned int fileSize;
		
// 	};
// #pragma pack(pop)	

// #pragma pack(push, 1)
// 	struct Texture2DHeader
// 	{
// 		unsigned short fileType;
// 		unsigned int fileSize;
// 		unsigned int width;
// 		unsigned int height;
// 		unsigned int numChannels;
// 		unsigned int sizeOfData;
// 	};
// #pragma pack(pop)	

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

	class Manager
	{
		private:
			// active entities and components
			int numberOfEntities;
			int numberOfTransforms;
			int numberOfRigidbodies;
			int numberOfMeshRenderers;
			int numberOfDirectionalLights;
			int numberOfSpotLights;
			int numberOfPointLights;

			// total number of entities and components allocated
			int totalNumberOfEntitiesAlloc;
			int totalNumberOfTransformsAlloc;
			int totalNumberOfRigidbodiesAlloc;
			int totalNumberOfMeshRenderersAlloc;
			int totalNumberOfDirectionalLightsAlloc;
			int totalNumberOfSpotLightsAlloc;
			int totalNumberOfPointLightsAlloc;

			std::map<int, int> idToIndexMap;
			std::map<int, int> componentIdToTypeMap;

			SceneSettings settings;

			// entities and components
			Entity* entities;
			Transform* transforms;
			Rigidbody* rigidbodies;
			MeshRenderer* meshRenderers;
			DirectionalLight* directionalLights;
			SpotLight* spotLights;
			PointLight* pointLights;

			// materials
			Material* materials;

			// shaders

			// textures

			// meshes
			float* vertices;
			float* normals;
			float* texCoords;

			// gmeshes


		public:
			Manager();
			~Manager();

		private:
			int loadAssets();
			int loadScene(const std::string &filepath);
	};
}

#endif


// TODO: So I could find all the asset files in win32_main and pass them through to scene which does not 
// store them at all and just immediately passes them through to manager? I.e. in win32_main just call 
// something like scene.loadMeshes(meshFilePaths) and then inside scene immediately call manager.loadMeshes(meshFilePaths)????
// In fact now that I think about it, is there any point to having the scene class at all??









































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