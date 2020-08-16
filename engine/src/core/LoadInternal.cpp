#include <iostream>

#include "../../include/core/LoadInternal.h"
#include "../../include/core/Log.h"

#include "../../include/systems/RenderSystem.h"
#include "../../include/systems/PhysicsSystem.h"
#include "../../include/systems/CleanUpSystem.h"
#include "../../include/systems/DebugSystem.h"

using namespace PhysicsEngine;

void PhysicsEngine::addInternalEntityIdToIndexMap(std::unordered_map<Guid, int>* entityIdToGlobalIndex, 
												  std::unordered_map<Guid, int>* idToGlobalIndex, 
												  Guid id, 
												  int index)
{
	(*entityIdToGlobalIndex)[id] = index;
	(*idToGlobalIndex)[id] = index;
}

void PhysicsEngine::addInternalComponentIdToIndexMap(std::unordered_map<Guid, int>* transformIdToGlobalIndex,
													std::unordered_map<Guid, int>* meshRendererIdToGlobalIndex,
													std::unordered_map<Guid, int>* lineRendererIdToGlobalIndex,
													std::unordered_map<Guid, int>* rigidbodyIdToGlobalIndex,
													std::unordered_map<Guid, int>* cameraIdToGlobalIndex,
													std::unordered_map<Guid, int>* lightIdToGlobalIndex,
													std::unordered_map<Guid, int>* sphereColliderIdToGlobalIndex,
													std::unordered_map<Guid, int>* boxColliderIdToGlobalIndex,
													std::unordered_map<Guid, int>* capsuleColliderIdToGlobalIndex,
													std::unordered_map<Guid, int>* meshColliderIdToGlobalIndex,
													std::unordered_map<Guid, int>* idToGlobalIndex,
													Guid id,
													int type,
													int index)
{
	if (type == ComponentType<Transform>::type) {
		(*transformIdToGlobalIndex)[id] = index;
		(*idToGlobalIndex)[id] = index;
	}
	else if (type == ComponentType<Rigidbody>::type) {
		(*rigidbodyIdToGlobalIndex)[id] = index;
		(*idToGlobalIndex)[id] = index;
	}
	else if (type == ComponentType<Camera>::type) {
		(*cameraIdToGlobalIndex)[id] = index;
		(*idToGlobalIndex)[id] = index;
	}
	else if (type == ComponentType<MeshRenderer>::type) {
		(*meshRendererIdToGlobalIndex)[id] = index;
		(*idToGlobalIndex)[id] = index;
	}
	else if (type == ComponentType<LineRenderer>::type) {
		(*lineRendererIdToGlobalIndex)[id] = index;
		(*idToGlobalIndex)[id] = index;
	}
	else if (type == ComponentType<Light>::type) {
		(*lightIdToGlobalIndex)[id] = index;
		(*idToGlobalIndex)[id] = index;
	}
	else if (type == ComponentType<BoxCollider>::type) {
		(*boxColliderIdToGlobalIndex)[id] = index;
		(*idToGlobalIndex)[id] = index;
	}
	else if (type == ComponentType<SphereCollider>::type) {
		(*sphereColliderIdToGlobalIndex)[id] = index;
		(*idToGlobalIndex)[id] = index;
	}
	else if (type == ComponentType<MeshCollider>::type) {
		(*meshColliderIdToGlobalIndex)[id] = index;
		(*idToGlobalIndex)[id] = index;
	}
	else if (type == ComponentType<CapsuleCollider>::type) {
		(*capsuleColliderIdToGlobalIndex)[id] = index;
		(*idToGlobalIndex)[id] = index;
	}
	else {
		std::string message = "Error: Invalid component type (" + std::to_string(type) + ") when trying to add internal component id to index map\n";
		Log::error(message.c_str());
	}
}

void PhysicsEngine::addInternalSystemIdToIndexMap(std::unordered_map<Guid, int>* renderSystemIdToGlobalIndex,
												std::unordered_map<Guid, int>* physicsSystemIdToGlobalIndex,
												std::unordered_map<Guid, int>* cleanupSystemIdToGlobalIndex,
												std::unordered_map<Guid, int>* debugSystemIdToGlobalIndex,
												std::unordered_map<Guid, int>* idToGlobalIndex,
												Guid id,
												int type,
												int index)
{
	if (type == SystemType<RenderSystem>::type) {
		(*renderSystemIdToGlobalIndex)[id] = index;
		(*idToGlobalIndex)[id] = index;
	}
	else if (type == SystemType<PhysicsSystem>::type) {
		(*physicsSystemIdToGlobalIndex)[id] = index;
		(*idToGlobalIndex)[id] = index;
	}
	else if (type == SystemType<CleanUpSystem>::type) {
		(*cleanupSystemIdToGlobalIndex)[id] = index;
		(*idToGlobalIndex)[id] = index;
	}
	else if (type == SystemType<DebugSystem>::type) {
		(*debugSystemIdToGlobalIndex)[id] = index;
		(*idToGlobalIndex)[id] = index;
	}
	else {
		std::string message = "Error: Invalid system type (" + std::to_string(type) + ") when trying to add internal system id to index map\n";
		Log::error(message.c_str());
	}
}

void PhysicsEngine::addInternalAssetIdToIndexMap(std::unordered_map<Guid, int>* meshIdToGlobalIndex,
												std::unordered_map<Guid, int>* materialIdToGlobalIndex,
												std::unordered_map<Guid, int>* shaderIdToGlobalIndex,
												std::unordered_map<Guid, int>* texture2DIdToGlobalIndex,
												std::unordered_map<Guid, int>* texture3DIdToGlobalIndex,
												std::unordered_map<Guid, int>* cubemapIdToGlobalIndex,
												std::unordered_map<Guid, int>* fontIdToGlobalIndex,
												std::unordered_map<Guid, int>* idToGlobalIndex,
												Guid id,
												int type,
												int index)
{
	if (type == AssetType<Shader>::type) {
		(*shaderIdToGlobalIndex)[id] = index;
		(*idToGlobalIndex)[id] = index;
	}
	else if (type == AssetType<Texture2D>::type) {
		(*texture2DIdToGlobalIndex)[id] = index;
		(*idToGlobalIndex)[id] = index;
	}
	else if (type == AssetType<Texture3D>::type) {
		(*texture3DIdToGlobalIndex)[id] = index;
		(*idToGlobalIndex)[id] = index;
	}
	else if (type == AssetType<Cubemap>::type) {
		(*cubemapIdToGlobalIndex)[id] = index;
		(*idToGlobalIndex)[id] = index;
	}
	else if (type == AssetType<Material>::type) {
		(*materialIdToGlobalIndex)[id] = index;
		(*idToGlobalIndex)[id] = index;
	}
	else if (type == AssetType<Mesh>::type) {
		(*meshIdToGlobalIndex)[id] = index;
		(*idToGlobalIndex)[id] = index;
	}
	else if (type == AssetType<Font>::type) {
		(*fontIdToGlobalIndex)[id] = index;
		(*idToGlobalIndex)[id] = index;
	}
	else {
		std::string message = "Error: Invalid asset type (" + std::to_string(type) + ") when trying to add internal asset id to index map\n";
		Log::error(message.c_str());
	}
}

Entity* PhysicsEngine::getInternalEntity(PoolAllocator<Entity>* entityAllocator, int index)
{
	return entityAllocator->get(index);
}

Component* PhysicsEngine::getInternalComponent(PoolAllocator<Transform>* transformAllocator,
	PoolAllocator<MeshRenderer>* meshRendererAllocator,
	PoolAllocator<LineRenderer>* lineRendererAllocator,
	PoolAllocator<Rigidbody>* rigidbodyAllocator,
	PoolAllocator<Camera>* cameraAllocator,
	PoolAllocator<Light>* lightAllocator,
	PoolAllocator<SphereCollider>* sphereColliderAllocator,
	PoolAllocator<BoxCollider>* boxColliderAllocator,
	PoolAllocator<CapsuleCollider>* capsuleColliderAllocator,
	PoolAllocator<MeshCollider>* meshColliderAllocator,
	int type,
	int index)
{
	if (type == ComponentType<Transform>::type) {
		return transformAllocator->get(index);
	}
	else if (type == ComponentType<Rigidbody>::type) {
		return rigidbodyAllocator->get(index);
	}
	else if (type == ComponentType<Camera>::type) {
		return cameraAllocator->get(index);
	}
	else if (type == ComponentType<MeshRenderer>::type) {
		return meshRendererAllocator->get(index);
	}
	else if (type == ComponentType<LineRenderer>::type) {
		return lineRendererAllocator->get(index);
	}
	else if (type == ComponentType<Light>::type) {
		return lightAllocator->get(index);
	}
	else if (type == ComponentType<BoxCollider>::type) {
		return boxColliderAllocator->get(index);
	}
	else if (type == ComponentType<SphereCollider>::type) {
		return sphereColliderAllocator->get(index);
	}
	else if (type == ComponentType<MeshCollider>::type) {
		return meshColliderAllocator->get(index);
	}
	else if (type == ComponentType<CapsuleCollider>::type) {
		return capsuleColliderAllocator->get(index);
	}
	else {
		std::string message = "Error: Invalid component type (" + std::to_string(type) + ") when trying to load internal component\n";
		Log::error(message.c_str());
		return NULL;
	}
}

System* PhysicsEngine::getInternalSystem(PoolAllocator<RenderSystem>* renderSystemAllocator,
	PoolAllocator<PhysicsSystem>* physicsSystemAllocator,
	PoolAllocator<CleanUpSystem>* cleanupSystemAllocator,
	PoolAllocator<DebugSystem>* debugSystemAllocator,
	int type,
	int index)
{
	if (type == SystemType<RenderSystem>::type) {
		return renderSystemAllocator->get(index);
	}
	else if (type == SystemType<PhysicsSystem>::type) {
		return physicsSystemAllocator->get(index);
	}
	else if (type == SystemType<CleanUpSystem>::type) {
		return cleanupSystemAllocator->get(index);
	}
	else if (type == SystemType<DebugSystem>::type) {
		return debugSystemAllocator->get(index);
	}
	else {
		std::string message = "Error: Invalid system type (" + std::to_string(type) + ") when trying to load internal system\n";
		Log::error(message.c_str());
		return NULL;
	}
}

Asset* PhysicsEngine::getInternalAsset(PoolAllocator<Mesh>* meshAllocator,
	PoolAllocator<Material>* materialAllocator,
	PoolAllocator<Shader>* shaderAllocator,
	PoolAllocator<Texture2D>* texture2DAllocator,
	PoolAllocator<Texture3D>* texture3DAllocator,
	PoolAllocator<Cubemap>* cubemapAllocator,
	PoolAllocator<Font>* fontAllocator,
	int type,
	int index)
{
	if (type == AssetType<Shader>::type) {
		return shaderAllocator->get(index);
	}
	else if (type == AssetType<Texture2D>::type) {
		return texture2DAllocator->get(index);
	}
	else if (type == AssetType<Texture3D>::type) {
		return texture3DAllocator->get(index);
	}
	else if (type == AssetType<Cubemap>::type) {
		return cubemapAllocator->get(index);
	}
	else if (type == AssetType<Material>::type) {
		return materialAllocator->get(index);
	}
	else if (type == AssetType<Mesh>::type) {
		return meshAllocator->get(index);
	}
	else if (type == AssetType<Font>::type) {
		return fontAllocator->get(index);
	}
	else {
		std::string message = "Error: Invalid asset type (" + std::to_string(type) + ") when trying to load internal asset\n";
		Log::error(message.c_str());
		return NULL;
	}
}

Entity* PhysicsEngine::loadInternalEntity(PoolAllocator<Entity>* entityAllocator, const std::vector<char>& data, int* index)
{
	*index = (int)entityAllocator->getCount();
	return entityAllocator->construct(data);
}

Component* PhysicsEngine::loadInternalComponent(PoolAllocator<Transform>* transformAllocator,
												PoolAllocator<MeshRenderer>* meshRendererAllocator,
												PoolAllocator<LineRenderer>* lineRendererAllocator,
												PoolAllocator<Rigidbody>* rigidbodyAllocator,
												PoolAllocator<Camera>* cameraAllocator,
												PoolAllocator<Light>* lightAllocator,
												PoolAllocator<SphereCollider>* sphereColliderAllocator,
												PoolAllocator<BoxCollider>* boxColliderAllocator,
												PoolAllocator<CapsuleCollider>* capsuleColliderAllocator,
												PoolAllocator<MeshCollider>* meshColliderAllocator,
												const std::vector<char>& data,
												int type,
												int* index)
{
	if (type == ComponentType<Transform>::type) {
		*index = (int)transformAllocator->getCount();
		return transformAllocator->construct(data);
	}
	else if (type == ComponentType<Rigidbody>::type) {
		*index = (int)rigidbodyAllocator->getCount();
		return rigidbodyAllocator->construct(data);
	}
	else if (type == ComponentType<Camera>::type) {
		*index = (int)cameraAllocator->getCount();
		return cameraAllocator->construct(data);
	}
	else if (type == ComponentType<MeshRenderer>::type) {
		*index = (int)meshRendererAllocator->getCount();
		return meshRendererAllocator->construct(data);
	}
	else if (type == ComponentType<LineRenderer>::type) {
		*index = (int)lineRendererAllocator->getCount();
		return lineRendererAllocator->construct(data);
	}
	else if (type == ComponentType<Light>::type) {
		*index = (int)lightAllocator->getCount();
		return lightAllocator->construct(data);
	}
	else if (type == ComponentType<BoxCollider>::type) {
		*index = (int)boxColliderAllocator->getCount();
		return boxColliderAllocator->construct(data);
	}
	else if (type == ComponentType<SphereCollider>::type) {
		*index = (int)sphereColliderAllocator->getCount();
		return sphereColliderAllocator->construct(data);
	}
	else if (type == ComponentType<MeshCollider>::type) {
		*index = (int)meshColliderAllocator->getCount();
		return meshColliderAllocator->construct(data);
	}
	else if (type == ComponentType<CapsuleCollider>::type) {
		*index = (int)capsuleColliderAllocator->getCount();
		return capsuleColliderAllocator->construct(data);
	}
	else {
		std::string message = "Error: Invalid component type (" + std::to_string(type) + ") when trying to load internal component\n";
		Log::error(message.c_str());
		return NULL;
	}
}

System* PhysicsEngine::loadInternalSystem(PoolAllocator<RenderSystem>* renderSystemAllocator,
										  PoolAllocator<PhysicsSystem>* physicsSystemAllocator,
										  PoolAllocator<CleanUpSystem>* cleanupSystemAllocator,
										  PoolAllocator<DebugSystem>* debugSystemAllocator,
										  const std::vector<char>& data,
										  int type,
										  int* index)
{
	if (type == SystemType<RenderSystem>::type) {
		*index = (int)renderSystemAllocator->getCount();
		return renderSystemAllocator->construct(data);
	}
	else if (type == SystemType<PhysicsSystem>::type) {
		*index = (int)physicsSystemAllocator->getCount();
		return physicsSystemAllocator->construct(data);
	}
	else if (type == SystemType<CleanUpSystem>::type) {
		*index = (int)cleanupSystemAllocator->getCount();
		return cleanupSystemAllocator->construct(data);
	}
	else if (type == SystemType<DebugSystem>::type) {
		*index = (int)debugSystemAllocator->getCount();
		return debugSystemAllocator->construct(data);
	}
	else {
		std::string message = "Error: Invalid system type (" + std::to_string(type) + ") when trying to load internal system\n";
		Log::error(message.c_str());
		return NULL;
	}
}

Asset* PhysicsEngine::loadInternalAsset(PoolAllocator<Mesh>* meshAllocator,
						PoolAllocator<Material>* materialAllocator,
						PoolAllocator<Shader>* shaderAllocator,
						PoolAllocator<Texture2D>* texture2DAllocator,
						PoolAllocator<Texture3D>* texture3DAllocator,
						PoolAllocator<Cubemap>* cubemapAllocator,
						PoolAllocator<Font>* fontAllocator,
						const std::vector<char>& data,
						int type,
						int* index)
{
	if (type == AssetType<Shader>::type) {
		*index = (int)shaderAllocator->getCount();
		return shaderAllocator->construct(data);
	}
	else if (type == AssetType<Texture2D>::type) {
		*index = (int)texture2DAllocator->getCount();
		return texture2DAllocator->construct(data);
	}
	else if (type == AssetType<Texture3D>::type) {
		*index = (int)texture3DAllocator->getCount();
		return texture3DAllocator->construct(data);
	}
	else if (type == AssetType<Cubemap>::type) {
		*index = (int)cubemapAllocator->getCount();
		return cubemapAllocator->construct(data);
	}
	else if (type == AssetType<Material>::type) {
		*index = (int)materialAllocator->getCount();
		return materialAllocator->construct(data);
	}
	else if (type == AssetType<Mesh>::type) {
		*index = (int)meshAllocator->getCount();
		return meshAllocator->construct(data);
	}
	else if (type == AssetType<Font>::type) {
		*index = (int)fontAllocator->getCount();
		return fontAllocator->construct(data);
	}
	else {
		std::string message = "Error: Invalid asset type (" + std::to_string(type) + ") when trying to load internal asset\n";
		Log::error(message.c_str());
		return NULL;
	}
}

Entity* PhysicsEngine::destroyInternalEntity(PoolAllocator<Entity>* entityAllocator, int index)
{
	return entityAllocator->destruct(index);
}

Component* PhysicsEngine::destroyInternalComponent(PoolAllocator<Transform>* transformAllocator,
									PoolAllocator<MeshRenderer>* meshRendererAllocator,
									PoolAllocator<LineRenderer>* lineRendererAllocator,
									PoolAllocator<Rigidbody>* rigidbodyAllocator,
									PoolAllocator<Camera>* cameraAllocator,
									PoolAllocator<Light>* lightAllocator,
									PoolAllocator<SphereCollider>* sphereColliderAllocator,
									PoolAllocator<BoxCollider>* boxColliderAllocator,
									PoolAllocator<CapsuleCollider>* capsuleColliderAllocator,
									PoolAllocator<MeshCollider>* meshColliderAllocator, int type, int index)
{
	if (type == ComponentType<Transform>::type) {
		return transformAllocator->destruct(index);
	}
	else if (type == ComponentType<Rigidbody>::type) {
		return rigidbodyAllocator->destruct(index);
	}
	else if (type == ComponentType<Camera>::type) {
		return cameraAllocator->destruct(index);
	}
	else if (type == ComponentType<MeshRenderer>::type) {
		return meshRendererAllocator->destruct(index);
	}
	else if (type == ComponentType<LineRenderer>::type) {
		return lineRendererAllocator->destruct(index);
	}
	else if (type == ComponentType<Light>::type) {
		return lightAllocator->destruct(index);
	}
	else if (type == ComponentType<BoxCollider>::type) {
		return boxColliderAllocator->destruct(index);
	}
	else if (type == ComponentType<SphereCollider>::type) {
		return sphereColliderAllocator->destruct(index);
	}
	else if (type == ComponentType<MeshCollider>::type) {
		return meshColliderAllocator->destruct(index);
	}
	else if (type == ComponentType<CapsuleCollider>::type) {
		return capsuleColliderAllocator->destruct(index);
	}
	else {
		std::string message = "Error: Invalid component instance type (" + std::to_string(type) + ") when trying to destroy internal component\n";
		Log::error(message.c_str());
		return NULL;
	}
}