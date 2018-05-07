#ifndef __MANAGER_H__
#define __MANAGER_H__

#include <vector>
#include <string>

#include "../entities/Entity.h"
#include "../components/Transform.h"
#include "../components/Rigidbody.h"
#include "../components/DirectionalLight.h"
#include "../components/PointLight.h"
#include "../components/SpotLight.h"
#include "../components/SphereCollider.h"
#include "../components/BoxCollider.h"
#include "../components/MeshRenderer.h"
#include "../components/LineRenderer.h"
#include "../components/Skybox.h"
#include "../components/Camera.h"
//#include "../components/Fluid.h"
//#include "../components/Cloth.h"

#include "Pool.h"

#include "../core/Mesh.h"
#include "../graphics/Texture2D.h"
#include "../graphics/Cubemap.h"
#include "../graphics/Shader.h"

namespace PhysicsEngine
{
	class Manager
	{
		private:
			// assets
			std::vector<Mesh> meshes;
			std::vector<Texture2D> textures;
			std::vector<Cubemap> cubemaps;
			std::vector<Shader> shaders;
			std::vector<Material> materials;

			std::map<std::string, int> meshMap;
			std::map<std::string, int> textureMap;
			std::map<std::string, int> cubemapMap;
			std::map<std::string, int> shaderMap;
			std::map<std::string, int> materialMap;

			std::vector<Entity*> entities;
			std::vector<Transform*> transforms;
			std::vector<Rigidbody*> rigidbodies;
			std::vector<DirectionalLight*> directionalLights;
			std::vector<PointLight*> pointLights;
			std::vector<SpotLight*> spotLights;
			std::vector<MeshRenderer*> meshRenderers;
			std::vector<LineRenderer*> lineRenderers;
			std::vector<SphereCollider*> sphereColliders;
			std::vector<BoxCollider*> boxColliders;
			std::vector<Collider*> colliders;
			//std::vector<Fluid*> fluids;
			//std::vector<Cloth*> cloths;

			Camera* camera;

			Pool<Entity> entityPool;
			Pool<Transform> transformPool;
			Pool<Rigidbody> rigidbodyPool;
			Pool<DirectionalLight> directionalLightPool;
			Pool<PointLight> pointLightPool;
			Pool<SpotLight> spotLightPool;
			Pool<MeshRenderer> meshRendererPool;
			Pool<LineRenderer> lineRendererPool;
			Pool<SphereCollider> sphereColliderPool;
			Pool<BoxCollider> boxColliderPool;
			Pool<Camera> cameraPool;
			//Pool<Fluid> fluidPool;
			//Pool<Cloth> clothPool;

		public:
			Manager();
			~Manager();

			Entity* createEntity();
			Transform* createTransform();
			Rigidbody* createRigidbody();
			DirectionalLight* createDirectionalLight();
			PointLight* createPointLight();
			SpotLight* createSpotLight();
			MeshRenderer* createMeshRenderer();
			LineRenderer* createLineRenderer();
			SphereCollider* createSphereCollider();
			BoxCollider* createBoxCollider();
			Camera* createCamera();
			//Fluid* createFluid();
			//Cloth* createCloth();

			std::vector<Entity*> getEntities();
			std::vector<Transform*> getTransforms();
			std::vector<Rigidbody*> getRigidbodies();
			std::vector<DirectionalLight*> getDirectionalLights();
			std::vector<PointLight*> getPointLights();
			std::vector<SpotLight*> getSpotLights();
			std::vector<MeshRenderer*> getMeshRenderers();
			std::vector<LineRenderer*> getLineRenderers();
			std::vector<SphereCollider*> getSphereColliders();
			std::vector<BoxCollider*> getBoxColliders();
			std::vector<Collider*> getColliders();
			//std::vector<Fluid*> getFluids();
			//std::vector<Cloth*> getCloths();
			Camera* getCamera();

			void loadMesh(const std::string& name);
			void loadTexture2D(const std::string& name);
			void loadCubemap(const std::vector<std::string>& names);
			void loadShader(const std::string& name, std::string vertex, std::string fragment, std::string geometry = std::string());
			void loadMaterial(const std::string& name, Material mat);

			Mesh* getMesh(const std::string& name);
			Texture2D* getTexture2D(const std::string& name);
			Cubemap* getCubemap(const std::string& name);
			Shader* getShader(const std::string& name);
			Material* getMaterial(const std::string& name);

			Mesh* getMesh(int filter);
			Material* getMaterial(int filter);

			std::vector<Mesh>& getMeshes();
			std::vector<Texture2D>& getTextures();
			std::vector<Cubemap>& getCubemaps();
			std::vector<Shader>& getShaders();
			std::vector<Material>& getMaterials();

			int getMeshFilter(const std::string& name);
			int getMaterialFilter(const std::string& name);
	};
}

#endif