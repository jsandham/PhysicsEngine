#include "Manager.h"
#include "../MeshLoader.h"
#include "../TextureLoader.h"

using namespace PhysicsEngine;

Manager::Manager()
{

}

Manager::~Manager()
{

}

Entity* Manager::createEntity()
{
	Entity *entity = entityPool.getNext();

	entities.push_back(entity);
	
	return entity;
}

//void Manager::destroyEntity(unsigned int index)
//{
//	if (index == entities.size()){ return; }
//
//	entityPool.swapWithLast(index);
//
//	// fix this components indicies now that it has been moved from the end to position index
//	//Entity *entity = entities[index];
//
//	entities.pop_back();
//}

Transform* Manager::createTransform()
{
	Transform* transform = transformPool.getNext();

	transforms.push_back(transform);

	return transform;
}

Rigidbody* Manager::createRigidbody()
{
	Rigidbody* rigidbody = rigidbodyPool.getNext();

	rigidbodies.push_back(rigidbody);

	return rigidbody;
}

DirectionalLight* Manager::createDirectionalLight()
{
	DirectionalLight* light = directionalLightPool.getNext();

	directionalLights.push_back(light);

	return light;
}

PointLight* Manager::createPointLight()
{
	PointLight* light = pointLightPool.getNext();

	pointLights.push_back(light);

	return light;
}

SpotLight* Manager::createSpotLight()
{
	SpotLight* light = spotLightPool.getNext();

	spotLights.push_back(light);

	return light;
}

MeshRenderer* Manager::createMeshRenderer()
{
	MeshRenderer* mesh = meshRendererPool.getNext();

	meshRenderers.push_back(mesh);

	return mesh;
}

LineRenderer* Manager::createLineRenderer()
{
	LineRenderer* line = lineRendererPool.getNext();

	lineRenderers.push_back(line);

	return line;
}

SphereCollider* Manager::createSphereCollider()
{
	SphereCollider* collider = sphereColliderPool.getNext();

	sphereColliders.push_back(collider);

	colliders.push_back(collider);

	return collider;
}

BoxCollider* Manager::createBoxCollider()
{
	BoxCollider* collider = boxColliderPool.getNext();

	boxColliders.push_back(collider);

	colliders.push_back(collider);

	return collider;
}

SpringJoint* Manager::createSpringJoint()
{
	SpringJoint* joint = springJointPool.getNext();

	springJoints.push_back(joint);

	joints.push_back(joint);

	return joint;
}

Fluid* Manager::createFluid()
{
	Fluid* fluid = fluidPool.getNext();

	fluids.push_back(fluid);

	return fluid;
}

Cloth* Manager::createCloth()
{
	Cloth* cloth = clothPool.getNext();
	cloths.push_back(cloth);

	return cloth;
}

Camera* Manager::createCamera()
{
	Camera* camera = cameraPool.getNext();

	this->camera = camera;

	return camera;
}

std::vector<Entity*> Manager::getEntities()
{
	return entities;
	//return entityPool.getPool();
}

std::vector<Transform*> Manager::getTransforms()
{
	return transforms;
	//return transformPool.getPool();
}

std::vector<Rigidbody*> Manager::getRigidbodies()
{
	return rigidbodies;
	//return rigidbodyPool.getPool();
}

std::vector<DirectionalLight*> Manager::getDirectionalLights()
{
	return directionalLights;
	//return directionalLightPool.getPool();
}

std::vector<PointLight*> Manager::getPointLights()
{
	return pointLights;
	//return pointLightPool.getPool();
}

std::vector<SpotLight*> Manager::getSpotLights()
{
	return spotLights;
	//return spotLightPool.getPool();
}

std::vector<MeshRenderer*> Manager::getMeshRenderers()
{
	return meshRenderers;
	//return meshRendererPool.getPool();
}

std::vector<LineRenderer*> Manager::getLineRenderers()
{
	return lineRenderers;
	//return lineRendererPool.getPool();
}

std::vector<Collider*> Manager::getColliders()
{
	return colliders;
}

std::vector<SphereCollider*> Manager::getSphereColliders()
{
	return sphereColliders;
}

std::vector<BoxCollider*> Manager::getBoxColliders()
{
	return boxColliders;
}

std::vector<Joint*> Manager::getJoints()
{
	return joints;
}

std::vector<SpringJoint*> Manager::getSpringJoints()
{
	return springJoints;
}

std::vector<Fluid*> Manager::getFluids()
{
	return fluids;
	//return fluidPool.getPool();
}

std::vector<Cloth*> Manager::getCloths()
{
	return cloths;
	//return clothPool.getPool();
}

Camera* Manager::getCamera()
{
	return camera;
	//return editorCameraPool.getPool();
}

void Manager::loadMesh(const std::string& name)
{
	if (meshMap.count(name) != 0){
		std::cout << "mesh: " << name << " already loaded" << std::endl;
		return;
	}

	std::cout << "loading mesh: " << name << std::endl;

	std::vector<float> vertices, normals, texCoords;

	if (MeshLoader::load(name, vertices, normals, texCoords)){
		Mesh mesh;
		mesh.setVertices(vertices);
		mesh.setNormals(normals);
		mesh.setTexCoords(texCoords);

		meshes.push_back(mesh);

		meshMap[name] = (int)meshes.size() - 1;
	}
	else{
		std::cout << "Could not load mesh " << name << std::endl;
	}
}

void Manager::loadTexture2D(const std::string& name)
{
	if (textureMap.count(name) != 0){
		std::cout << "texture: " << name << " already loaded" << std::endl;
		return;
	}
	int width, height, numChannels;
	std::vector<unsigned char> rawTextureData;
	
	if (TextureLoader::load(name, rawTextureData, &width, &height, &numChannels)){
		
		TextureFormat format = Red;
		if (numChannels == 3){
			format = TextureFormat::RGB;
		}
		else if (numChannels == 4){
			format = TextureFormat::RGBA;
		}
		else{
			std::cout << "Manager: Number of channels not supported" << std::endl;
			return;
		}

		Texture2D texture(width, height, format);
		texture.setRawTextureData(rawTextureData);

		textures.push_back(texture);

		textureMap[name] = (int)textures.size() - 1;
	}
	else{
		std::cout << "Could not load texture " << name << std::endl;
	}
}

void Manager::loadCubemap(const std::vector<std::string>& names)
{
	std::string name;
	for (unsigned int i = 0; i < names.size(); i++){
		name += names[i];
	}

	if (cubemapMap.count(name) != 0){
		std::cout << "Manager: Cubemap texture: " << name << " already loaded" << std::endl;
		return;
	}

	if (names.size() != 6){
		std::cout << "Manager: When loading cubemaps, exactly 6 filenames must be passed" << std::endl;
		return;
	}

	int width, height, numChannels;
	std::vector<unsigned char> rawCubemapData;

	for (unsigned int i = 0; i < 6; i++){
		std::vector<unsigned char> data;

		if (!TextureLoader::load(names[i], data, &width, &height, &numChannels)){
			std::cout << "Manager: Could not load " << i << "th image of cubemap " << names[i] << std::endl;
			return;
		}

		for (unsigned int j = 0; j < data.size(); j++){
			rawCubemapData.push_back(data[j]);
		}
	}

	if (rawCubemapData.size() != 6*width*height*numChannels){
		std::cout << "Manager: each face of the cubemap must have the same size and channels" << std::endl;
		return;
	}

	TextureFormat format = Red;
	if (numChannels == 3){
		format = TextureFormat::RGB;
	}
	else if (numChannels == 4){
		format = TextureFormat::RGBA;
	}
	else{
		std::cout << "Manager: Number of channels not supported" << std::endl;
		return;
	}

	Cubemap cubemap(width, format);

	cubemap.setRawCubemapData(rawCubemapData);

	cubemaps.push_back(cubemap);

	cubemapMap[name] = (int)cubemaps.size() - 1;
}

void Manager::loadShader(const std::string& name, std::string vertex, std::string fragment, std::string geometry)
{
	if (shaderMap.count(name) != 0){
		std::cout << "shader program: " << name << " already loaded" << std::endl;
		return;
	}

	std::cout << "loading shader program: " << name << std::endl;

	shaders.push_back(Shader(vertex, fragment, geometry));

	shaderMap[name] = (int)shaders.size() - 1;
}

void Manager::loadMaterial(const std::string& name, Material mat)
{
	if (materialMap.count(name) != 0){
		std::cout << "material " << name << " already loaded" << std::endl;
		return;
	}

	std::cout << "loading material: " << name << std::endl;

	materials.push_back(mat);

	materialMap[name] = (int)materials.size() - 1;
}

Mesh* Manager::getMesh(const std::string& name)
{
	std::map<std::string, int>::iterator it = meshMap.find(name);
	if (it != meshMap.end()){
		return &meshes[it->second];
	}

	return NULL;
}

Texture2D* Manager::getTexture2D(const std::string& name)
{
	std::map<std::string, int>::iterator it = textureMap.find(name);
	if (it != textureMap.end()){
		return &textures[it->second];
	}

	return NULL;
}

Cubemap* Manager::getCubemap(const std::string& name)
{
	std::map<std::string, int>::iterator it = cubemapMap.find(name);
	if (it != cubemapMap.end()){
		return &cubemaps[it->second];
	}

	return NULL;
}

Shader* Manager::getShader(const std::string& name)
{
	std::map<std::string, int>::iterator it = shaderMap.find(name);
	if (it != shaderMap.end()){
		return &shaders[it->second];
	}

	return NULL;
}

Material* Manager::getMaterial(const std::string& name)
{
	std::map<std::string, int>::iterator it = materialMap.find(name);
	if (it != materialMap.end()){
		return &materials[it->second];
	}

	return NULL;
}

std::vector<Mesh>& Manager::getMeshes()
{
	return meshes;
}

std::vector<Texture2D>& Manager::getTextures()
{
	return textures;
}

std::vector<Cubemap>& Manager::getCubemaps()
{
	return cubemaps;
}

std::vector<Shader>& Manager::getShaders()
{
	return shaders;
}

std::vector<Material>& Manager::getMaterials()
{
	return materials;
}

int Manager::getMeshFilter(const std::string& name)
{
	std::map<std::string, int>::iterator it = meshMap.find(name);
	if (it != meshMap.end()){
		return it->second;
	}

	return -1;
}

int Manager::getMaterialFilter(const std::string& name)
{
	std::map<std::string, int>::iterator it = materialMap.find(name);
	if (it != materialMap.end()){
		return it->second;
	}

	return -1;
}

Mesh* Manager::getMesh(int filter)
{
	if (filter < 0 || filter >= (int)meshes.size()){
		std::cout << "Invalid mesh filter: " << filter << std::endl;
	}

	return &meshes[filter];
}

Material* Manager::getMaterial(int filter)
{
	if (filter < 0 || filter >= (int)materials.size()){
		std::cout << "Invalid material filter: " << filter << std::endl;
	}

	return &materials[filter];
}