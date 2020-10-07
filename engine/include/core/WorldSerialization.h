#ifndef __WORLD_SERIALIZATION_H__
#define __WORLD_SERIALIZATION_H__

#include <string>
#include <unordered_map>

#include "PoolAllocator.h"

#include "Cubemap.h"
#include "Font.h"
#include "Material.h"
#include "Mesh.h"
#include "Shader.h"
#include "Texture2D.h"
#include "Texture3D.h"

#include "../components/BoxCollider.h"
#include "../components/Camera.h"
#include "../components/CapsuleCollider.h"
#include "../components/Light.h"
#include "../components/LineRenderer.h"
#include "../components/MeshCollider.h"
#include "../components/MeshRenderer.h"
#include "../components/Rigidbody.h"
#include "../components/SphereCollider.h"
#include "../components/Transform.h"

#include "../systems/CleanUpSystem.h"
#include "../systems/DebugSystem.h"
#include "../systems/PhysicsSystem.h"
#include "../systems/RenderSystem.h"

#include "Util.h"
#include "WorldUtil.h"

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

void loadAssetIntoWorld(const std::string &filepath, WorldAllocators &allocators, WorldIsState &idState,
                        std::unordered_map<Guid, std::string> &assetIdToFilepath);

void loadSceneIntoWorld(const std::string &filepath, WorldAllocators &allocators, WorldIsState &idState,
                        std::unordered_map<Guid, std::string> &sceneIdToFilepath);
} // namespace PhysicsEngine

#endif