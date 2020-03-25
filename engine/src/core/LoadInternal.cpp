#include <iostream>

#include "../../include/core/LoadInternal.h"
#include "../../include/core/PoolAllocator.h"
#include "../../include/core/Entity.h"
#include "../../include/core/Shader.h"
#include "../../include/core/Texture2D.h"
#include "../../include/core/Texture3D.h"
#include "../../include/core/Cubemap.h"
#include "../../include/core/Material.h"
#include "../../include/core/Mesh.h"
#include "../../include/core/Font.h"
#include "../../include/core/Log.h"

#include "../../include/components/Transform.h"
#include "../../include/components/Rigidbody.h"
#include "../../include/components/Camera.h"
#include "../../include/components/MeshRenderer.h"
#include "../../include/components/LineRenderer.h"
#include "../../include/components/Light.h"
#include "../../include/components/BoxCollider.h"
#include "../../include/components/SphereCollider.h"
#include "../../include/components/MeshCollider.h"
#include "../../include/components/CapsuleCollider.h"

#include "../../include/systems/RenderSystem.h"
#include "../../include/systems/PhysicsSystem.h"
#include "../../include/systems/CleanUpSystem.h"
#include "../../include/systems/DebugSystem.h"

using namespace PhysicsEngine;

Entity* PhysicsEngine::loadInternalEntity(Allocator* allocator, std::vector<char> data, int* index)
{
	if (allocator == NULL) {
		allocator = new PoolAllocator<Entity>();
	}

	PoolAllocator<Entity>* poolAllocator = static_cast<PoolAllocator<Entity>*>(allocator);

	*index = (int)poolAllocator->getCount();
	return poolAllocator->construct(data);
}

Component* PhysicsEngine::loadInternalComponent(std::map<int, Allocator*>* allocatorMap, std::vector<char> data, int type, int* index)
{
	if (allocatorMap == NULL) {
		return NULL;
	}

	Allocator* allocator = NULL;

	std::map<int, Allocator*>::iterator it = allocatorMap->find(type);
	if (it != allocatorMap->end()) {
		allocator = it->second;
	}

	if (type == 0) {
		if (allocator == NULL) {
			allocator = new PoolAllocator<Transform>();
			(*allocatorMap)[type] = allocator;
		}

		PoolAllocator<Transform>* poolAllocator = static_cast<PoolAllocator<Transform>*>(allocator);

		*index = (int)poolAllocator->getCount();
		return poolAllocator->construct(data);
	}
	else if (type == 1) {
		if (allocator == NULL) {
			allocator = new PoolAllocator<Rigidbody>();
			(*allocatorMap)[type] = allocator;
		}

		PoolAllocator<Rigidbody>* poolAllocator = static_cast<PoolAllocator<Rigidbody>*>(allocator);

		*index = (int)poolAllocator->getCount();
		return poolAllocator->construct(data);
	}
	else if (type == 2) {
		if (allocator == NULL) {
			allocator = new PoolAllocator<Camera>();
			(*allocatorMap)[type] = allocator;
		}

		PoolAllocator<Camera>* poolAllocator = static_cast<PoolAllocator<Camera>*>(allocator);

		*index = (int)poolAllocator->getCount();
		return poolAllocator->construct(data);
	}
	else if (type == 3) {
		if (allocator == NULL) {
			allocator = new PoolAllocator<MeshRenderer>();
			(*allocatorMap)[type] = allocator;
		}

		PoolAllocator<MeshRenderer>* poolAllocator = static_cast<PoolAllocator<MeshRenderer>*>(allocator);

		*index = (int)poolAllocator->getCount();
		return poolAllocator->construct(data);
	}
	else if (type == 4) {
		if (allocator == NULL) {
			allocator = new PoolAllocator<LineRenderer>();
			(*allocatorMap)[type] = allocator;
		}

		PoolAllocator<LineRenderer>* poolAllocator = static_cast<PoolAllocator<LineRenderer>*>(allocator);

		*index = (int)poolAllocator->getCount();
		return poolAllocator->construct(data);
	}
	else if (type == 5) {
		if (allocator == NULL) {
			allocator = new PoolAllocator<Light>();
			(*allocatorMap)[type] = allocator;
		}

		PoolAllocator<Light>* poolAllocator = static_cast<PoolAllocator<Light>*>(allocator);

		*index = (int)poolAllocator->getCount();
		return poolAllocator->construct(data);
	}
	else if (type == 8) {
		if (allocator == NULL) {
			allocator = new PoolAllocator<BoxCollider>();
			(*allocatorMap)[type] = allocator;
		}

		PoolAllocator<BoxCollider>* poolAllocator = static_cast<PoolAllocator<BoxCollider>*>(allocator);

		*index = (int)poolAllocator->getCount();
		return poolAllocator->construct(data);
	}
	else if (type == 9) {
		if (allocator == NULL) {
			allocator = new PoolAllocator<SphereCollider>();
			(*allocatorMap)[type] = allocator;
		}

		PoolAllocator<SphereCollider>* poolAllocator = static_cast<PoolAllocator<SphereCollider>*>(allocator);

		*index = (int)poolAllocator->getCount();
		return poolAllocator->construct(data);
	}
	else if (type == 10) {
		if (allocator == NULL) {
			allocator = new PoolAllocator<MeshCollider>();
			(*allocatorMap)[type] = allocator;
		}

		PoolAllocator<MeshCollider>* poolAllocator = static_cast<PoolAllocator<MeshCollider>*>(allocator);

		*index = (int)poolAllocator->getCount();
		return poolAllocator->construct(data);
	}
	else if (type == 15) {
		if (allocator == NULL) {
			allocator = new PoolAllocator<CapsuleCollider>();
			(*allocatorMap)[type] = allocator;
		}

		PoolAllocator<CapsuleCollider>* poolAllocator = static_cast<PoolAllocator<CapsuleCollider>*>(allocator);

		*index = (int)poolAllocator->getCount();
		return poolAllocator->construct(data);
	}
	else {
		std::string message = "Error: Invalid component type (" + std::to_string(type) + ") when trying to load internal component\n";
		Log::error(message.c_str());
		return NULL;
	}
}

System* PhysicsEngine::loadInternalSystem(std::map<int, Allocator*>* allocatorMap, std::vector<char> data, int type, int* index)
{
	if (allocatorMap == NULL) {
		return NULL;
	}

	Allocator* allocator = NULL;

	std::map<int, Allocator*>::iterator it = allocatorMap->find(type);
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

Asset* PhysicsEngine::loadInternalAsset(std::map<int, Allocator*>* allocatorMap, std::vector<char> data, int type, int* index)
{
	if (allocatorMap == NULL) {
		return NULL;
	}

	Allocator* allocator = NULL;

	std::map<int, Allocator*>::iterator it = allocatorMap->find(type);
	if (it != allocatorMap->end()) {
		allocator = it->second;
	}

	if (type == 0) {
		if (allocator == NULL) {
			allocator = new PoolAllocator<Shader>();
			(*allocatorMap)[type] = allocator;
		}

		PoolAllocator<Shader>* poolAllocator = static_cast<PoolAllocator<Shader>*>(allocator);

		*index = (int)poolAllocator->getCount();
		return poolAllocator->construct(data);
	}
	else if (type == 1) {
		if (allocator == NULL) {
			allocator = new PoolAllocator<Texture2D>();
			(*allocatorMap)[type] = allocator;
		}

		PoolAllocator<Texture2D>* poolAllocator = static_cast<PoolAllocator<Texture2D>*>(allocator);

		*index = (int)poolAllocator->getCount();
		return poolAllocator->construct(data);
	}
	else if (type == 2) {
		if (allocator == NULL) {
			allocator = new PoolAllocator<Texture3D>();
			(*allocatorMap)[type] = allocator;
		}

		PoolAllocator<Texture3D>* poolAllocator = static_cast<PoolAllocator<Texture3D>*>(allocator);

		*index = (int)poolAllocator->getCount();
		return poolAllocator->construct(data);
	}
	else if (type == 3) {
		if (allocator == NULL) {
			allocator = new PoolAllocator<Cubemap>();
			(*allocatorMap)[type] = allocator;
		}

		PoolAllocator<Cubemap>* poolAllocator = static_cast<PoolAllocator<Cubemap>*>(allocator);

		*index = (int)poolAllocator->getCount();
		return poolAllocator->construct(data);
	}
	else if (type == 4) {
		if (allocator == NULL) {
			allocator = new PoolAllocator<Material>();
			(*allocatorMap)[type] = allocator;
		}

		PoolAllocator<Material>* poolAllocator = static_cast<PoolAllocator<Material>*>(allocator);

		*index = (int)poolAllocator->getCount();
		return poolAllocator->construct(data);
	}
	else if (type == 5) {
		if (allocator == NULL) {
			allocator = new PoolAllocator<Mesh>();
			(*allocatorMap)[type] = allocator;
		}

		PoolAllocator<Mesh>* poolAllocator = static_cast<PoolAllocator<Mesh>*>(allocator);

		*index = (int)poolAllocator->getCount();
		return poolAllocator->construct(data);
	}
	else if (type == 6) {
		if (allocator == NULL) {
			allocator = new PoolAllocator<Font>();
			(*allocatorMap)[type] = allocator;
		}

		PoolAllocator<Font>* poolAllocator = static_cast<PoolAllocator<Font>*>(allocator);

		*index = (int)poolAllocator->getCount();
		return poolAllocator->construct(data);
	}
	else {
		std::string message = "Error: Invalid asset type (" + std::to_string(type) + ") when trying to load internal asset\n";
		Log::error(message.c_str());
		return NULL;
	}
}

Entity* PhysicsEngine::destroyInternalEntity(Allocator* allocator, int index)
{
	if (allocator == NULL) {
		return NULL;
	}

	PoolAllocator<Entity>* poolAllocator = static_cast<PoolAllocator<Entity>*>(allocator);

	return poolAllocator->destruct(index);
}

Component* PhysicsEngine::destroyInternalComponent(std::map<int, Allocator*>* allocatorMap, int type, int index)
{
	if (allocatorMap == NULL) {
		return NULL;
	}

	Allocator* allocator = NULL;

	std::map<int, Allocator*>::iterator it = allocatorMap->find(type);
	if (it != allocatorMap->end()) {
		allocator = it->second;
	}

	if (allocator == NULL) {
		return NULL;
	}

	if (type == ComponentType<Transform>::type) {
		PoolAllocator<Transform>* poolAllocator = static_cast<PoolAllocator<Transform>*>(allocator);

		return poolAllocator->destruct(index);
	}
	else if (type == ComponentType<Rigidbody>::type) {
		PoolAllocator<Rigidbody>* poolAllocator = static_cast<PoolAllocator<Rigidbody>*>(allocator);

		return poolAllocator->destruct(index);
	}
	else if (type == ComponentType<Camera>::type) {
		PoolAllocator<Camera>* poolAllocator = static_cast<PoolAllocator<Camera>*>(allocator);

		return poolAllocator->destruct(index);
	}
	else if (type == ComponentType<MeshRenderer>::type) {
		PoolAllocator<MeshRenderer>* poolAllocator = static_cast<PoolAllocator<MeshRenderer>*>(allocator);

		return poolAllocator->destruct(index);
	}
	else if (type == ComponentType<LineRenderer>::type) {
		PoolAllocator<LineRenderer>* poolAllocator = static_cast<PoolAllocator<LineRenderer>*>(allocator);

		return poolAllocator->destruct(index);
	}
	else if (type == ComponentType<Light>::type) {
		PoolAllocator<Light>* poolAllocator = static_cast<PoolAllocator<Light>*>(allocator);

		return poolAllocator->destruct(index);
	}
	else if (type == ComponentType<BoxCollider>::type) {
		PoolAllocator<BoxCollider>* poolAllocator = static_cast<PoolAllocator<BoxCollider>*>(allocator);

		return poolAllocator->destruct(index);
	}
	else if (type == ComponentType<SphereCollider>::type) {
		PoolAllocator<SphereCollider>* poolAllocator = static_cast<PoolAllocator<SphereCollider>*>(allocator);

		return poolAllocator->destruct(index);
	}
	else if (type == ComponentType<MeshCollider>::type) {
		PoolAllocator<MeshCollider>* poolAllocator = static_cast<PoolAllocator<MeshCollider>*>(allocator);

		return poolAllocator->destruct(index);
	}
	else if (type == ComponentType<CapsuleCollider>::type) {
		PoolAllocator<CapsuleCollider>* poolAllocator = static_cast<PoolAllocator<CapsuleCollider>*>(allocator);

		return poolAllocator->destruct(index);
	}
	else {
		std::string message = "Error: Invalid component instance type (" + std::to_string(type) + ") when trying to destroy internal component\n";
		Log::error(message.c_str());
		return NULL;
	}
}