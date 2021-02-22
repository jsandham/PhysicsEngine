#ifndef __WORLD_ALLOCATORS_H__
#define __WORLD_ALLOCATORS_H__

#include <unordered_map>

#include "Allocator.h"
#include "Guid.h"
#include "PoolAllocator.h"

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

#include "../core/Cubemap.h"
#include "../core/Font.h"
#include "../core/Material.h"
#include "../core/Mesh.h"
#include "../core/Texture2D.h"
#include "../core/Texture3D.h"

#include "../systems/CleanUpSystem.h"
#include "../systems/DebugSystem.h"
#include "../systems/GizmoSystem.h"
#include "../systems/PhysicsSystem.h"
#include "../systems/RenderSystem.h"

namespace PhysicsEngine
{
// Simple structs used for grouping world allocators when passing to functions
struct WorldAllocators
{
    // internal entity allocator
    PoolAllocator<Entity> mEntityAllocator;

    // internal component allocators
    PoolAllocator<Transform> mTransformAllocator;
    PoolAllocator<MeshRenderer> mMeshRendererAllocator;
    PoolAllocator<LineRenderer> mLineRendererAllocator;
    PoolAllocator<Rigidbody> mRigidbodyAllocator;
    PoolAllocator<Camera> mCameraAllocator;
    PoolAllocator<Light> mLightAllocator;
    PoolAllocator<SphereCollider> mSphereColliderAllocator;
    PoolAllocator<BoxCollider> mBoxColliderAllocator;
    PoolAllocator<CapsuleCollider> mCapsuleColliderAllocator;
    PoolAllocator<MeshCollider> mMeshColliderAllocator;

    // internal asset allocators
    PoolAllocator<Mesh> mMeshAllocator;
    PoolAllocator<Material> mMaterialAllocator;
    PoolAllocator<Shader> mShaderAllocator;
    PoolAllocator<Texture2D> mTexture2DAllocator;
    PoolAllocator<Texture3D> mTexture3DAllocator;
    PoolAllocator<Cubemap> mCubemapAllocator;
    PoolAllocator<Font> mFontAllocator;

    // internal system allocators
    PoolAllocator<RenderSystem> mRenderSystemAllocator;
    PoolAllocator<PhysicsSystem> mPhysicsSystemAllocator;
    PoolAllocator<CleanUpSystem> mCleanupSystemAllocator;
    PoolAllocator<DebugSystem> mDebugSystemAllocator;
    PoolAllocator<GizmoSystem> mGizmoSystemAllocator;

    // non-internal allocators for user defined components, systems and assets
    std::unordered_map<int, Allocator *> mComponentAllocatorMap;
    std::unordered_map<int, Allocator *> mSystemAllocatorMap;
    std::unordered_map<int, Allocator *> mAssetAllocatorMap;

    ~WorldAllocators()
    {
        for (auto it = mComponentAllocatorMap.begin(); it != mComponentAllocatorMap.end(); it++)
        {
            delete it->second;
        }

        for (auto it = mSystemAllocatorMap.begin(); it != mSystemAllocatorMap.end(); it++)
        {
            delete it->second;
        }

        for (auto it = mAssetAllocatorMap.begin(); it != mAssetAllocatorMap.end(); it++)
        {
            delete it->second;
        }
    }
};
} // namespace PhysicsEngine

#endif