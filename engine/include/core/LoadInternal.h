#ifndef __LOADINTERNAL_H__
#define __LOADINTERNAL_H__

#include <vector>
#include <unordered_map>

#include "Allocator.h"
#include "PoolAllocator.h"

#include "Entity.h"

#include "Asset.h"
#include "Mesh.h"
#include "Material.h"
#include "Shader.h"
#include "Texture2D.h"
#include "Texture3D.h"
#include "Cubemap.h"
#include "Font.h"

#include "../components/Component.h"
#include "../components/Transform.h"
#include "../components/MeshRenderer.h"
#include "../components/Camera.h"
#include "../components/Light.h"
#include "../components/Rigidbody.h"
#include "../components/LineRenderer.h"
#include "../components/SphereCollider.h"
#include "../components/BoxCollider.h"
#include "../components/CapsuleCollider.h"
#include "../components/MeshCollider.h"

#include "../systems/System.h"

namespace PhysicsEngine
{
	Entity* loadInternalEntity(PoolAllocator<Entity>* entityAllocator, std::vector<char> data, int* index);
	Component* loadInternalComponent(PoolAllocator<Transform>* transformAllocator,
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
									 int* index);
	System* loadInternalSystem(std::unordered_map<int, Allocator*>* allocatorMap, std::vector<char> data, int type, int* index);
	Asset* loadInternalAsset(PoolAllocator<Mesh>* meshAllocator,
							 PoolAllocator<Material>* materialAllocator,
							 PoolAllocator<Shader>* shaderAllocator,
							 PoolAllocator<Texture2D>* texture2DAllocator,
							 PoolAllocator<Texture3D>* texture3DAllocator,
							 PoolAllocator<Cubemap>* cubemapAllocator,
							 PoolAllocator<Font>* fontAllocator,
							 std::vector<char> data, 
							 int type, 
							 int* index);

	Entity* destroyInternalEntity(PoolAllocator<Entity>* entityAllocator, int index);
	Component* destroyInternalComponent(PoolAllocator<Transform>* transformAllocator,
										PoolAllocator<MeshRenderer>* meshRendererAllocator,
										PoolAllocator<LineRenderer>* lineRendererAllocator,
										PoolAllocator<Rigidbody>* rigidbodyAllocator,
										PoolAllocator<Camera>* cameraAllocator,
										PoolAllocator<Light>* lightAllocator,
										PoolAllocator<SphereCollider>* sphereColliderAllocator,
										PoolAllocator<BoxCollider>* boxColliderAllocator,
										PoolAllocator<CapsuleCollider>* capsuleColliderAllocator,
										PoolAllocator<MeshCollider>* meshColliderAllocator, int type, int index);

	/*Entity* loadInternalEntity(Allocator* allocator, std::vector<char> data, int* index);
	Component* loadInternalComponent(std::unordered_map<int, Allocator*>* allocatorMap, std::vector<char> data, int type, int* index);
	System* loadInternalSystem(std::unordered_map<int, Allocator*>* allocatorMap, std::vector<char> data, int type, int* index);
	Asset* loadInternalAsset(std::unordered_map<int, Allocator*>* allocatorMap, std::vector<char> data, int type, int* index);

	Entity* destroyInternalEntity(Allocator* allocator, int index);
	Component* destroyInternalComponent(std::unordered_map<int, Allocator*>* allocatorMap, int type, int index);*/
}

#endif