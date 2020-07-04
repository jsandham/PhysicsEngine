#include <iostream>

#include "../../include/core/LoadInternal.h"
#include "../../include/core/Log.h"

#include "../../include/systems/RenderSystem.h"
#include "../../include/systems/PhysicsSystem.h"
#include "../../include/systems/CleanUpSystem.h"
#include "../../include/systems/DebugSystem.h"

using namespace PhysicsEngine;

Entity* PhysicsEngine::loadInternalEntity(PoolAllocator<Entity>* entityAllocator, std::vector<char> data, int* index)
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
												std::vector<char> data,
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
										  std::vector<char> data,
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
						std::vector<char> data,
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