#ifndef WORLD_ALLOCATORS_H__
#define WORLD_ALLOCATORS_H__

#include <unordered_map>

#include "Guid.h"
#include "PoolAllocator.h"

#include "../core/Scene.h"

#include "../core/Cubemap.h"
#include "../core/Font.h"
#include "../core/Material.h"
#include "../core/Mesh.h"
#include "../core/Sprite.h"
#include "../core/Texture2D.h"
#include "../core/Texture3D.h"

#include "../systems/CleanUpSystem.h"
#include "../systems/DebugSystem.h"
#include "../systems/GizmoSystem.h"
#include "../systems/PhysicsSystem.h"
#include "../systems/RenderSystem.h"
#include "../systems/FreeLookCameraSystem.h"
#include "../systems/TerrainSystem.h"

namespace PhysicsEngine
{
// Simple structs used for grouping world allocators when passing to functions
struct WorldAllocators
{
    // internal scene allocator
    PoolAllocator<Scene> mSceneAllocator;

    // internal asset allocators
    PoolAllocator<Mesh> mMeshAllocator;
    PoolAllocator<Material> mMaterialAllocator;
    PoolAllocator<Shader> mShaderAllocator;
    PoolAllocator<Texture2D> mTexture2DAllocator;
    PoolAllocator<Texture3D> mTexture3DAllocator;
    PoolAllocator<Cubemap> mCubemapAllocator;
    PoolAllocator<RenderTexture> mRenderTextureAllocator;
    PoolAllocator<Font> mFontAllocator;
    PoolAllocator<Sprite> mSpriteAllocator;

    // internal system allocators
    PoolAllocator<RenderSystem> mRenderSystemAllocator;
    PoolAllocator<PhysicsSystem> mPhysicsSystemAllocator;
    PoolAllocator<CleanUpSystem> mCleanupSystemAllocator;
    PoolAllocator<DebugSystem> mDebugSystemAllocator;
    PoolAllocator<GizmoSystem> mGizmoSystemAllocator;
    PoolAllocator<FreeLookCameraSystem> mFreeLookCameraSystemAllocator;
    PoolAllocator<TerrainSystem> mTerrainSystemAllocator;
};
} // namespace PhysicsEngine

#endif