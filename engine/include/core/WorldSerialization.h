#ifndef __WORLD_SERIALIZATION_H__
#define __WORLD_SERIALIZATION_H__

#include <string>
#include <unordered_map>

#include "PoolAllocator.h"

#include "Mesh.h"
#include "Material.h"
#include "Shader.h"
#include "Texture2D.h"
#include "Texture3D.h"
#include "Cubemap.h"
#include "Font.h"

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

#include "../systems/RenderSystem.h"
#include "../systems/PhysicsSystem.h"
#include "../systems/CleanUpSystem.h"
#include "../systems/DebugSystem.h"

#include "Util.h"

namespace PhysicsEngine
{
	extern const uint64_t ASSET_FILE_SIGNATURE;
	extern const uint64_t SCENE_FILE_SIGNATURE;

#pragma pack(push, 1)
	struct AssetFileHeader
	{
		Guid mAssetId;
		uint64_t mSignature;
		int32_t mType;
		size_t mSize;
		uint8_t mMajor;
		uint8_t mMinor;
	};
#pragma pack(pop)

#pragma pack(push, 1)
	struct SceneFileHeader
	{
		uint64_t mSignature;
		Guid mSceneId;
		size_t mSize;
		int32_t mEntityCount;
		int32_t mComponentCount;
		int32_t mSystemCount;
		uint8_t mMajor;
		uint8_t mMinor;
	};
#pragma pack(pop)

#pragma pack(push, 1)
	struct ComponentInfoHeader
	{
		Guid mComponentId;
		int32_t mType;
		size_t mStartPtr;
		size_t mSize;
	};
#pragma pack(pop)

#pragma pack(push, 1)
	struct SystemInfoHeader
	{
		Guid mSystemId;
		int32_t mType;
		size_t mStartPtr;
		size_t mSize;
	};
#pragma pack(pop)

	void loadAssetIntoWorld(const std::string& filepath,
							PoolAllocator<Mesh>& meshAllocator,
							PoolAllocator<Material>& materialAllocator,
							PoolAllocator<Shader>& shaderAllocator,
							PoolAllocator<Texture2D>& texture2DAllocator,
							PoolAllocator<Texture3D>& texture3DAllocator,
							PoolAllocator<Cubemap>& cubemapAllocator,
							PoolAllocator<Font>& fontAllocator,
							std::unordered_map<int, Allocator*>& assetAllocatorMap,
							std::unordered_map<Guid, int>& idToGlobalIndex, 
							std::unordered_map<Guid, int>& idToType,
							std::unordered_map<Guid, std::string>& assetIdToFilepath);

	void loadSceneIntoWorld(const std::string& filepath,
							PoolAllocator<Entity>& entityAllocator,
							PoolAllocator<Transform>& transformAllocator,
							PoolAllocator<MeshRenderer>& meshRendererAllocator,
							PoolAllocator<LineRenderer>& lineRendererAllocator,
							PoolAllocator<Rigidbody>& rigidbodyAllocator,
							PoolAllocator<Camera>& cameraAllocator,
							PoolAllocator<Light>& lightAllocator,
							PoolAllocator<SphereCollider>& sphereColliderAllocator,
							PoolAllocator<BoxCollider>& boxColliderAllocator,
							PoolAllocator<CapsuleCollider>& capsuleColliderAllocator,
							PoolAllocator<MeshCollider>& meshColliderAllocator,
							PoolAllocator<RenderSystem>& renderSystemAllocator,
							PoolAllocator<PhysicsSystem>& physicsSystemAllocator,
							PoolAllocator<CleanUpSystem>& cleanupSystemAllocator,
							PoolAllocator<DebugSystem>& debugSystemAllocator,
							std::unordered_map<int, Allocator*>& componentAllocatorMap,
							std::unordered_map<int, Allocator*>& systemAllocatorMap,
							std::unordered_map<Guid, int>& idToGlobalIndex,
							std::unordered_map<Guid, int>& idToType,
							std::unordered_map<Guid, std::vector<std::pair<Guid, int>>>& entityIdToComponentIds,
							std::vector<Guid>& entityIdsMarkedCreated,
							std::vector<triple<Guid, Guid, int>>& componentIdsMarkedCreated,
							std::unordered_map<Guid, std::string>& sceneIdToFilepath);
}

#endif