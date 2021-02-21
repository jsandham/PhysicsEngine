#ifndef __WORLD_SERIALIZATION_H__
#define __WORLD_SERIALIZATION_H__

#include "Guid.h"

namespace PhysicsEngine
{
    //extern const uint64_t ASSET_FILE_SIGNATURE;
    //extern const uint64_t SCENE_FILE_SIGNATURE;

    // in cpp file
    //const uint64_t PhysicsEngine::ASSET_FILE_SIGNATURE = 0x9a9e9b4153534554;
    //const uint64_t PhysicsEngine::SCENE_FILE_SIGNATURE = 0x9a9e9b5343454e45;

    const uint64_t ASSET_FILE_SIGNATURE = 0x9a9e9b4153534554;
    const uint64_t SCENE_FILE_SIGNATURE = 0x9a9e9b5343454e45;

#pragma pack(push, 1)
    struct AssetHeader
    {
        Guid mAssetId;
        uint64_t mSignature;
        int32_t mType;
        size_t mSize;
        int32_t mAssetCount;
        uint8_t mMajor;
        uint8_t mMinor;
    };
#pragma pack(pop)

#pragma pack(push, 1)
    struct SceneHeader
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
    struct ObjectHeader
    {
        Guid mId;
        int32_t mType;
        bool mIsTnternal;
    };
#pragma pack(pop)
}

#endif