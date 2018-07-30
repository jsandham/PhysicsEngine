#include <iostream>

#include "Manager.h"

using namespace PhysicsEngine;

Manager::Manager()
{
	entities = NULL;
	transforms = NULL;
	rigidbodies = NULL;
	meshRenderers = NULL;
	directionalLights = NULL;
	spotLights = NULL;
	pointLights = NULL;
}

Manager::~Manager()
{
	delete [] entities;
	delete [] transforms;
	delete [] rigidbodies;
	delete [] meshRenderers;
	delete [] directionalLights;
	delete [] spotLights;
	delete [] pointLights;
}

int Manager::loadAssets( )
{
	return 0;
}

int Manager::loadScene(const std::string &filepath)
{
	SceneHeader sceneHeader = {};
	FILE* file = fopen(filepath.c_str(), "rb");
	size_t bytesRead;
	if (file){
		bytesRead = fread(&sceneHeader, sizeof(SceneHeader), 1, file);
		std::cout << "number of bytes read from file: " << bytesRead << std::endl;
	}
	else{
		std::cout << "Error: Failed to open file " << filepath << " for reading" << std::endl;
		return 0;
	}

	std::cout << "de-serialized scene header file contains the following information: " << std::endl;
	std::cout << "fileSize: " << sceneHeader.fileSize << std::endl;

	std::cout << "numberOfEntities: " << sceneHeader.numberOfEntities << std::endl;
	std::cout << "numberOfTransforms: " << sceneHeader.numberOfTransforms << std::endl;
	std::cout << "numberOfRigidbodies: " << sceneHeader.numberOfRigidbodies << std::endl;
	std::cout << "numberOfMeshRenderers: " << sceneHeader.numberOfMeshRenderers << std::endl;
	std::cout << "numberOfDirectionalLights: " << sceneHeader.numberOfDirectionalLights << std::endl;
	std::cout << "numberOfSpotLights: " << sceneHeader.numberOfSpotLights << std::endl;
	std::cout << "numberOfPointLights: " << sceneHeader.numberOfPointLights << std::endl;

	std::cout << "sizeOfEntity: " << sceneHeader.sizeOfEntity << std::endl;
	std::cout << "sizeOfTransform: " << sceneHeader.sizeOfTransform << std::endl;
	std::cout << "sizeOfRigidbodies: " << sceneHeader.sizeOfRigidbody << std::endl;
	std::cout << "sizeOfMeshRenderer: " << sceneHeader.sizeOfMeshRenderer << std::endl;
	std::cout << "sizeOfDirectionalLight: " << sceneHeader.sizeOfDirectionalLight << std::endl;
	std::cout << "sizeOfSpotLight: " << sceneHeader.sizeOfSpotLight << std::endl;
	std::cout << "sizeOfPointLight: " << sceneHeader.sizeOfPointLight << std::endl;

	numberOfEntities = sceneHeader.numberOfEntities;
	numberOfTransforms = sceneHeader.numberOfTransforms;
	numberOfRigidbodies = sceneHeader.numberOfRigidbodies;
	numberOfMeshRenderers = sceneHeader.numberOfMeshRenderers;
	numberOfDirectionalLights = sceneHeader.numberOfDirectionalLights;
	numberOfSpotLights = sceneHeader.numberOfSpotLights;
	numberOfPointLights = sceneHeader.numberOfPointLights;

	bytesRead = fread(&settings, sizeof(SceneSettings), 1, file);

	totalNumberOfEntitiesAlloc = settings.maxAllowedEntities;
	totalNumberOfTransformsAlloc = settings.maxAllowedTransforms;
	totalNumberOfRigidbodiesAlloc = settings.maxAllowedRigidbodies;
	totalNumberOfMeshRenderersAlloc = settings.maxAllowedMeshRenderers;
	totalNumberOfDirectionalLightsAlloc= settings.maxAllowedDirectionalLights;
	totalNumberOfSpotLightsAlloc = settings.maxAllowedSpotLights;
	totalNumberOfPointLightsAlloc = settings.maxAllowedPointLights;

	bool error = numberOfEntities > totalNumberOfEntitiesAlloc;
	error &= numberOfTransforms > totalNumberOfTransformsAlloc;
	error &= numberOfRigidbodies > totalNumberOfRigidbodiesAlloc;
	error &= numberOfMeshRenderers > totalNumberOfMeshRenderersAlloc;
	error &= numberOfDirectionalLights > totalNumberOfDirectionalLightsAlloc;
	error &= numberOfSpotLights > totalNumberOfSpotLightsAlloc;
	error &= numberOfPointLights > totalNumberOfPointLightsAlloc;

	if(error){
		std::cout << "Error: Number of entities or components exceeds maximum allowed. Please increase max allowed in scene settings." << std::endl;
		return 0;
	}

	// allocate memory blocks for entities and components
	entities = new Entity[totalNumberOfEntitiesAlloc];
	transforms = new Transform[totalNumberOfTransformsAlloc];
	rigidbodies = new Rigidbody[totalNumberOfRigidbodiesAlloc];
	meshRenderers = new MeshRenderer[totalNumberOfMeshRenderersAlloc];
	directionalLights = new DirectionalLight[totalNumberOfDirectionalLightsAlloc];
	spotLights = new SpotLight[totalNumberOfSpotLightsAlloc];
	pointLights = new PointLight[totalNumberOfPointLightsAlloc];

	// de-serialize entities and components
	bytesRead = fread(entities, sizeof(Entity), 1, file);
	bytesRead = fread(transforms, sizeof(Transform), 1, file);
	bytesRead = fread(rigidbodies, sizeof(Rigidbody), 1, file);
	bytesRead = fread(meshRenderers, sizeof(MeshRenderer), 1, file);
	bytesRead = fread(directionalLights, sizeof(DirectionalLight), 1, file);
	bytesRead = fread(spotLights, sizeof(SpotLight), 1, file);
	bytesRead = fread(pointLights, sizeof(PointLight), 1, file);

	// map entity/component id to its global array index
	for(int i = 0; i < numberOfEntities; i++){ idToIndexMap[entities[i].entityId] = i; }
	for(int i = 0; i < numberOfTransforms; i++){ idToIndexMap[transforms[i].componentId] = i; }
	for(int i = 0; i < numberOfRigidbodies; i++){ idToIndexMap[rigidbodies[i].componentId] = i; }
	for(int i = 0; i < numberOfMeshRenderers; i++){ idToIndexMap[meshRenderers[i].componentId] = i; }
	for(int i = 0; i < numberOfDirectionalLights; i++){ idToIndexMap[directionalLights[i].componentId] = i; }
	for(int i = 0; i < numberOfSpotLights; i++){ idToIndexMap[spotLights[i].componentId] = i; }
	for(int i = 0; i < numberOfPointLights; i++){ idToIndexMap[pointLights[i].componentId] = i; }

	// map component id to its type
	for(int i = 0; i < numberOfTransforms; i++){ componentIdToTypeMap[transforms[i].componentId] = (int)ComponentType::TransformType; }
	for(int i = 0; i < numberOfRigidbodies; i++){ componentIdToTypeMap[rigidbodies[i].componentId] = (int)ComponentType::RigidbodyType; }
	for(int i = 0; i < numberOfMeshRenderers; i++){ componentIdToTypeMap[meshRenderers[i].componentId] = (int)ComponentType::MeshRendererType; }
	for(int i = 0; i < numberOfDirectionalLights; i++){ componentIdToTypeMap[directionalLights[i].componentId] = (int)ComponentType::DirectionalLightType; }
	for(int i = 0; i < numberOfSpotLights; i++){ componentIdToTypeMap[spotLights[i].componentId] = (int)ComponentType::SpotLightType; }
	for(int i = 0; i < numberOfPointLights; i++){ componentIdToTypeMap[pointLights[i].componentId] = (int)ComponentType::PointLightType; }

	// set global indices in entities and components
	for(int i = 0; i < numberOfEntities; i++){
		entities[i].globalEntityIndex = i;

		for(int j = 0; j < 8; j++){
			int componentId = entities[i].componentIds[j];

			int globalComponentIndex = idToIndexMap.find(componentId)->second;
			int componentType = componentIdToTypeMap.find(componentId)->second;

			entities[i].globalComponentIndices[j] = globalComponentIndex;
			entities[i].componentTypes[j] = componentType;
		}
	}

	for(int i = 0; i < numberOfTransforms; i++){
		transforms[i].globalComponentIndex = i;

		int entityId = transforms[i].entityId;
		int globalEntityIndex = idToIndexMap.find(entityId)->second;
		transforms[i].globalEntityIndex = globalEntityIndex;
	}

	fclose(file);

	return 1;
}

























































// #include "Manager.h"
// #include "../MeshLoader.h"
// #include "../TextureLoader.h"

// using namespace PhysicsEngine;

// Manager::Manager()
// {

// }

// Manager::~Manager()
// {

// }

// Entity* Manager::createEntity()
// {
// 	Entity *entity = entityPool.getNext();

// 	entity->globalEntityIndex = (int)entities.size();
// 	entities.push_back(entity);
	
// 	return entity;
// }

// //void Manager::destroyEntity(unsigned int index)
// //{
// //	if (index == entities.size()){ return; }
// //
// //	entityPool.swapWithLast(index);
// //
// //	// fix this components indicies now that it has been moved from the end to position index
// //	//Entity *entity = entities[index];
// //
// //	entities.pop_back();
// //}

// Transform* Manager::createTransform()
// {
// 	Transform* transform = transformPool.getNext();

// 	transform->globalComponentIndex = (int)transforms.size();
// 	transforms.push_back(transform);

// 	return transform;
// }

// Rigidbody* Manager::createRigidbody()
// {
// 	Rigidbody* rigidbody = rigidbodyPool.getNext();

// 	rigidbody->globalComponentIndex = (int)rigidbodies.size();
// 	rigidbodies.push_back(rigidbody);

// 	return rigidbody;
// }

// DirectionalLight* Manager::createDirectionalLight()
// {
// 	DirectionalLight* light = directionalLightPool.getNext();

// 	light->globalComponentIndex = (int)directionalLights.size();
// 	directionalLights.push_back(light);

// 	return light;
// }

// PointLight* Manager::createPointLight()
// {
// 	PointLight* light = pointLightPool.getNext();

// 	light->globalComponentIndex = (int)pointLights.size();
// 	pointLights.push_back(light);

// 	return light;
// }

// SpotLight* Manager::createSpotLight()
// {
// 	SpotLight* light = spotLightPool.getNext();

// 	light->globalComponentIndex = (int)spotLights.size();
// 	spotLights.push_back(light);

// 	return light;
// }

// MeshRenderer* Manager::createMeshRenderer()
// {
// 	MeshRenderer* mesh = meshRendererPool.getNext();

// 	mesh->globalComponentIndex = (int)meshRenderers.size();
// 	meshRenderers.push_back(mesh);

// 	return mesh;
// }

// // LineRenderer* Manager::createLineRenderer()
// // {
// // 	LineRenderer* line = lineRendererPool.getNext();

// // 	line->globalComponentIndex = (int)lineRenderers.size();
// // 	lineRenderers.push_back(line);

// // 	return line;
// // }

// SphereCollider* Manager::createSphereCollider()
// {
// 	SphereCollider* collider = sphereColliderPool.getNext();

// 	collider->globalComponentIndex = (int)sphereColliders.size();
// 	sphereColliders.push_back(collider);

// 	colliders.push_back(collider);

// 	return collider;
// }

// BoxCollider* Manager::createBoxCollider()
// {
// 	BoxCollider* collider = boxColliderPool.getNext();

// 	collider->globalComponentIndex = (int)boxColliders.size();
// 	boxColliders.push_back(collider);

// 	colliders.push_back(collider);

// 	return collider;
// }

// SpringJoint* Manager::createSpringJoint()
// {
// 	SpringJoint* joint = springJointPool.getNext();

// 	joint->globalComponentIndex = (int)springJoints.size();
// 	springJoints.push_back(joint);

// 	joints.push_back(joint);

// 	return joint;
// }

// Fluid* Manager::createFluid()
// {
// 	Fluid* fluid = fluidPool.getNext();

// 	fluid->globalComponentIndex = (int)fluids.size();
// 	fluids.push_back(fluid);

// 	return fluid;
// }

// Cloth* Manager::createCloth()
// {
// 	Cloth* cloth = clothPool.getNext();

// 	cloth->globalComponentIndex = (int)cloths.size();
// 	cloths.push_back(cloth);

// 	return cloth;
// }

// Solid* Manager::createSolid()
// {
// 	Solid* solid = solidPool.getNext();

// 	solid->globalComponentIndex = (int)solids.size();
// 	solids.push_back(solid);

// 	return solid;
// }

// Camera* Manager::createCamera()
// {
// 	Camera* camera = cameraPool.getNext();

// 	this->camera = camera;

// 	return camera;
// }

// std::vector<Entity*> Manager::getEntities()
// {
// 	return entities;
// 	//return entityPool.getPool();
// }

// std::vector<Transform*> Manager::getTransforms()
// {
// 	return transforms;
// 	//return transformPool.getPool();
// }

// std::vector<Rigidbody*> Manager::getRigidbodies()
// {
// 	return rigidbodies;
// 	//return rigidbodyPool.getPool();
// }

// std::vector<DirectionalLight*> Manager::getDirectionalLights()
// {
// 	return directionalLights;
// 	//return directionalLightPool.getPool();
// }

// std::vector<PointLight*> Manager::getPointLights()
// {
// 	return pointLights;
// 	//return pointLightPool.getPool();
// }

// std::vector<SpotLight*> Manager::getSpotLights()
// {
// 	return spotLights;
// 	//return spotLightPool.getPool();
// }

// std::vector<MeshRenderer*> Manager::getMeshRenderers()
// {
// 	return meshRenderers;
// 	//return meshRendererPool.getPool();
// }

// // std::vector<LineRenderer*> Manager::getLineRenderers()
// // {
// // 	return lineRenderers;
// // 	//return lineRendererPool.getPool();
// // }

// std::vector<Collider*> Manager::getColliders()
// {
// 	return colliders;
// }

// std::vector<SphereCollider*> Manager::getSphereColliders()
// {
// 	return sphereColliders;
// }

// std::vector<BoxCollider*> Manager::getBoxColliders()
// {
// 	return boxColliders;
// }

// std::vector<Joint*> Manager::getJoints()
// {
// 	return joints;
// }

// std::vector<SpringJoint*> Manager::getSpringJoints()
// {
// 	return springJoints;
// }

// std::vector<Fluid*> Manager::getFluids()
// {
// 	return fluids;
// 	//return fluidPool.getPool();
// }

// std::vector<Cloth*> Manager::getCloths()
// {
// 	return cloths;
// 	//return clothPool.getPool();
// }

// std::vector<Solid*> Manager::getSolids()
// {
// 	return solids;
// 	//return clothPool.getPool();
// }

// Camera* Manager::getCamera()
// {
// 	return camera;
// 	//return editorCameraPool.getPool();
// }

// void Manager::loadGMesh(const std::string& name)
// {
// 	if (gmeshMap.count(name) != 0){
// 		std::cout << "gmesh: " << name << " already loaded" << std::endl;
// 		return;
// 	}

// 	std::cout << "loading gmesh: " << name << std::endl;

// 	GMesh gmesh;
// 	if (MeshLoader::load_gmesh(name, gmesh)){
// 		gmeshes.push_back(gmesh);

// 		gmeshMap[name] = (int)gmeshes.size() - 1;
// 	}
// 	else{
// 		std::cout << "Could not load gmesh " << name << std::endl;
// 	}
// }

// void Manager::loadMesh(const std::string& name)
// {
// 	if (meshMap.count(name) != 0){
// 		std::cout << "mesh: " << name << " already loaded" << std::endl;
// 		return;
// 	}

// 	std::cout << "loading mesh: " << name << std::endl;

// 	Mesh mesh;
// 	if (MeshLoader::load(name, mesh)){
// 		meshes.push_back(mesh);

// 		meshMap[name] = (int)meshes.size() - 1;
// 	}
// 	else{
// 		std::cout << "Could not load mesh " << name << std::endl;
// 	}
// }

// void Manager::loadTexture2D(const std::string& name)
// {
// 	if (textureMap.count(name) != 0){
// 		std::cout << "texture: " << name << " already loaded" << std::endl;
// 		return;
// 	}
// 	int width, height, numChannels;
// 	std::vector<unsigned char> rawTextureData;
	
// 	if (TextureLoader::load(name, rawTextureData, &width, &height, &numChannels)){
		
// 		TextureFormat format = Red;
// 		if (numChannels == 3){
// 			format = TextureFormat::RGB;
// 		}
// 		else if (numChannels == 4){
// 			format = TextureFormat::RGBA;
// 		}
// 		else{
// 			std::cout << "Manager: Number of channels not supported" << std::endl;
// 			return;
// 		}

// 		Texture2D texture(width, height, format);
// 		texture.setRawTextureData(rawTextureData);

// 		textures.push_back(texture);

// 		textureMap[name] = (int)textures.size() - 1;
// 	}
// 	else{
// 		std::cout << "Could not load texture " << name << std::endl;
// 	}
// }

// void Manager::loadCubemap(const std::vector<std::string>& names)
// {
// 	std::string name;
// 	for (unsigned int i = 0; i < names.size(); i++){
// 		name += names[i];
// 	}

// 	if (cubemapMap.count(name) != 0){
// 		std::cout << "Manager: Cubemap texture: " << name << " already loaded" << std::endl;
// 		return;
// 	}

// 	if (names.size() != 6){
// 		std::cout << "Manager: When loading cubemaps, exactly 6 filenames must be passed" << std::endl;
// 		return;
// 	}

// 	int width, height, numChannels;
// 	std::vector<unsigned char> rawCubemapData;

// 	for (unsigned int i = 0; i < 6; i++){
// 		std::vector<unsigned char> data;

// 		if (!TextureLoader::load(names[i], data, &width, &height, &numChannels)){
// 			std::cout << "Manager: Could not load " << i << "th image of cubemap " << names[i] << std::endl;
// 			return;
// 		}

// 		for (unsigned int j = 0; j < data.size(); j++){
// 			rawCubemapData.push_back(data[j]);
// 		}
// 	}

// 	if (rawCubemapData.size() != 6*width*height*numChannels){
// 		std::cout << "Manager: each face of the cubemap must have the same size and channels" << std::endl;
// 		return;
// 	}

// 	TextureFormat format = Red;
// 	if (numChannels == 3){
// 		format = TextureFormat::RGB;
// 	}
// 	else if (numChannels == 4){
// 		format = TextureFormat::RGBA;
// 	}
// 	else{
// 		std::cout << "Manager: Number of channels not supported" << std::endl;
// 		return;
// 	}

// 	Cubemap cubemap(width, format);

// 	cubemap.setRawCubemapData(rawCubemapData);

// 	cubemaps.push_back(cubemap);

// 	cubemapMap[name] = (int)cubemaps.size() - 1;
// }

// void Manager::loadShader(const std::string& name, std::string vertex, std::string fragment, std::string geometry)
// {
// 	if (shaderMap.count(name) != 0){
// 		std::cout << "shader program: " << name << " already loaded" << std::endl;
// 		return;
// 	}

// 	std::cout << "loading shader program: " << name << std::endl;

// 	shaders.push_back(Shader(vertex, fragment, geometry));

// 	shaderMap[name] = (int)shaders.size() - 1;
// }

// void Manager::loadMaterial(const std::string& name, Material mat)
// {
// 	if (materialMap.count(name) != 0){
// 		std::cout << "material " << name << " already loaded" << std::endl;
// 		return;
// 	}

// 	std::cout << "loading material: " << name << std::endl;

// 	materials.push_back(mat);

// 	materialMap[name] = (int)materials.size() - 1;
// }

// GMesh* Manager::getGMesh(const std::string& name)
// {
// 	std::map<std::string, int>::iterator it = gmeshMap.find(name);
// 	if (it != gmeshMap.end()){
// 		return &gmeshes[it->second];
// 	}

// 	return NULL;
// }

// Mesh* Manager::getMesh(const std::string& name)
// {
// 	std::map<std::string, int>::iterator it = meshMap.find(name);
// 	if (it != meshMap.end()){
// 		return &meshes[it->second];
// 	}

// 	return NULL;
// }

// Texture2D* Manager::getTexture2D(const std::string& name)
// {
// 	std::map<std::string, int>::iterator it = textureMap.find(name);
// 	if (it != textureMap.end()){
// 		return &textures[it->second];
// 	}

// 	return NULL;
// }

// Cubemap* Manager::getCubemap(const std::string& name)
// {
// 	std::map<std::string, int>::iterator it = cubemapMap.find(name);
// 	if (it != cubemapMap.end()){
// 		return &cubemaps[it->second];
// 	}

// 	return NULL;
// }

// Shader* Manager::getShader(const std::string& name)
// {
// 	std::map<std::string, int>::iterator it = shaderMap.find(name);
// 	if (it != shaderMap.end()){
// 		return &shaders[it->second];
// 	}

// 	return NULL;
// }

// Material* Manager::getMaterial(const std::string& name)
// {
// 	std::map<std::string, int>::iterator it = materialMap.find(name);
// 	if (it != materialMap.end()){
// 		return &materials[it->second];
// 	}

// 	return NULL;
// }

// std::vector<GMesh>& Manager::getGMeshes()
// {
// 	return gmeshes;
// }

// std::vector<Mesh>& Manager::getMeshes()
// {
// 	return meshes;
// }

// std::vector<Texture2D>& Manager::getTextures()
// {
// 	return textures;
// }

// std::vector<Cubemap>& Manager::getCubemaps()
// {
// 	return cubemaps;
// }

// std::vector<Shader>& Manager::getShaders()
// {
// 	return shaders;
// }

// std::vector<Material>& Manager::getMaterials()
// {
// 	return materials;
// }

// int Manager::getGMeshFilter(const std::string& name)
// {
// 	std::map<std::string, int>::iterator it = gmeshMap.find(name);
// 	if (it != gmeshMap.end()){
// 		return it->second;
// 	}

// 	return -1;
// }

// int Manager::getMeshFilter(const std::string& name)
// {
// 	std::map<std::string, int>::iterator it = meshMap.find(name);
// 	if (it != meshMap.end()){
// 		return it->second;
// 	}

// 	return -1;
// }

// int Manager::getMaterialFilter(const std::string& name)
// {
// 	std::map<std::string, int>::iterator it = materialMap.find(name);
// 	if (it != materialMap.end()){
// 		return it->second;
// 	}

// 	return -1;
// }

// GMesh* Manager::getGMesh(int filter)
// {
// 	if (filter < 0 || filter >= (int)gmeshes.size()){
// 		std::cout << "Invalid gmesh filter: " << filter << std::endl;
// 	}

// 	return &gmeshes[filter];
// }

// Mesh* Manager::getMesh(int filter)
// {
// 	if (filter < 0 || filter >= (int)meshes.size()){
// 		std::cout << "Invalid mesh filter: " << filter << std::endl;
// 	}

// 	return &meshes[filter];
// }

// Material* Manager::getMaterial(int filter)
// {
// 	if (filter < 0 || filter >= (int)materials.size()){
// 		std::cout << "Invalid material filter: " << filter << std::endl;
// 	}

// 	return &materials[filter];
// }