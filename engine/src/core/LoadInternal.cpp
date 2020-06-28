#include <iostream>

#include "../../include/core/LoadInternal.h"

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
	if (type == 0) {
		*index = (int)transformAllocator->getCount();
		return transformAllocator->construct(data);
	}
	else if (type == 1) {
		*index = (int)rigidbodyAllocator->getCount();
		return rigidbodyAllocator->construct(data);
	}
	else if (type == 2) {
		*index = (int)cameraAllocator->getCount();
		return cameraAllocator->construct(data);
	}
	else if (type == 3) {
		*index = (int)meshRendererAllocator->getCount();
		return meshRendererAllocator->construct(data);
	}
	else if (type == 4) {
		*index = (int)lineRendererAllocator->getCount();
		return lineRendererAllocator->construct(data);
	}
	else if (type == 5) {
		*index = (int)lightAllocator->getCount();
		return lightAllocator->construct(data);
	}
	else if (type == 8) {
		*index = (int)boxColliderAllocator->getCount();
		return boxColliderAllocator->construct(data);
	}
	else if (type == 9) {
		*index = (int)sphereColliderAllocator->getCount();
		return sphereColliderAllocator->construct(data);
	}
	else if (type == 10) {
		*index = (int)meshColliderAllocator->getCount();
		return meshColliderAllocator->construct(data);
	}
	else if (type == 15) {
		*index = (int)capsuleColliderAllocator->getCount();
		return capsuleColliderAllocator->construct(data);
	}
	else {
		std::string message = "Error: Invalid component type (" + std::to_string(type) + ") when trying to load internal component\n";
		Log::error(message.c_str());
		return NULL;
	}
}

System* PhysicsEngine::loadInternalSystem(std::unordered_map<int, Allocator*>* allocatorMap, std::vector<char> data, int type, int* index)
{
	if (allocatorMap == NULL) {
		return NULL;
	}

	Allocator* allocator = NULL;

	std::unordered_map<int, Allocator*>::iterator it = allocatorMap->find(type);
	if (it != allocatorMap->end()) {
		allocator = it->second;
	}

	if (type == 0) {
		if (allocator == NULL) {
			allocator = new PoolAllocator<RenderSystem>();
			(*allocatorMap)[type] = allocator;
		}

		PoolAllocator<RenderSystem>* poolAllocator = static_cast<PoolAllocator<RenderSystem>*>(allocator);

		*index = (int)poolAllocator->getCount();
		return poolAllocator->construct(data);
	}
	else if (type == 1) {
		if (allocator == NULL) {
			allocator = new PoolAllocator<PhysicsSystem>();
			(*allocatorMap)[type] = allocator;
		}

		PoolAllocator<PhysicsSystem>* poolAllocator = static_cast<PoolAllocator<PhysicsSystem>*>(allocator);

		*index = (int)poolAllocator->getCount();
		return poolAllocator->construct(data);
	}
	else if (type == 2) {
		if (allocator == NULL) {
			allocator = new PoolAllocator<CleanUpSystem>();
			(*allocatorMap)[type] = allocator;
		}

		PoolAllocator<CleanUpSystem>* poolAllocator = static_cast<PoolAllocator<CleanUpSystem>*>(allocator);

		*index = (int)poolAllocator->getCount();
		return poolAllocator->construct(data);
	}
	else if (type == 3) {
		if (allocator == NULL) {
			allocator = new PoolAllocator<DebugSystem>();
			(*allocatorMap)[type] = allocator;
		}

		PoolAllocator<DebugSystem>* poolAllocator = static_cast<PoolAllocator<DebugSystem>*>(allocator);

		*index = (int)poolAllocator->getCount();
		return poolAllocator->construct(data);
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
	if (type == 0) {
		*index = (int)shaderAllocator->getCount();
		return shaderAllocator->construct(data);
	}
	else if (type == 1) {
		*index = (int)texture2DAllocator->getCount();
		return texture2DAllocator->construct(data);
	}
	else if (type == 2) {
		*index = (int)texture3DAllocator->getCount();
		return texture3DAllocator->construct(data);
	}
	else if (type == 3) {
		*index = (int)cubemapAllocator->getCount();
		return cubemapAllocator->construct(data);
	}
	else if (type == 4) {
		*index = (int)materialAllocator->getCount();
		return materialAllocator->construct(data);
	}
	else if (type == 5) {
		*index = (int)meshAllocator->getCount();
		return meshAllocator->construct(data);
	}
	else if (type == 6) {
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