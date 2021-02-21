#include <iostream>

#include "../../include/core/LoadInternal.h"
#include "../../include/core/Log.h"

#include "../../include/systems/CleanUpSystem.h"
#include "../../include/systems/DebugSystem.h"
#include "../../include/systems/PhysicsSystem.h"
#include "../../include/systems/RenderSystem.h"

using namespace PhysicsEngine;

Entity* PhysicsEngine::loadInternalEntity(WorldAllocators& allocators, WorldIdState& state, std::istream& in, const Guid& id)
{
    std::unordered_map<Guid, int>::iterator it = state.mEntityIdToGlobalIndex.find(id);
    if (it != state.mEntityIdToGlobalIndex.end()) {
        Entity* entity = allocators.mEntityAllocator.get(it->second);

        if (entity != nullptr) {
            entity->deserialize(in);

            return entity;
        }
    }

    int index = (int)allocators.mEntityAllocator.getCount();
    Entity* entity = allocators.mEntityAllocator.construct(in);

    if (entity != nullptr) {
        state.mEntityIdToGlobalIndex[entity->getId()] = index;
        state.mIdToGlobalIndex[entity->getId()] = index;
        state.mIdToType[entity->getId()] = EntityType<Entity>::type;
    }

    return entity;
}

Component* loadInternalComponent(WorldAllocators& allocators, WorldIdState& state, std::istream& in, const Guid& id, int type)
{
    if (type == ComponentType<Transform>::type)
    {
        std::unordered_map<Guid, int>::iterator it = state.mTransformIdToGlobalIndex.find(id);
        if (it != state.mTransformIdToGlobalIndex.end()) {
            Component* component = allocators.mTransformAllocator.get(it->second);

            if (component != nullptr) {
                component->deserialize(in);

                return component;
            }
        }
    }
    else if (type == ComponentType<Rigidbody>::type)
    {
        std::unordered_map<Guid, int>::iterator it = state.mRigidbodyIdToGlobalIndex.find(id);
        if (it != state.mRigidbodyIdToGlobalIndex.end()) {
            Component* component = allocators.mRigidbodyAllocator.get(it->second);

            if (component != nullptr) {
                component->deserialize(in);

                return component;
            }
        }
    }
    else if (type == ComponentType<Camera>::type)
    {
        std::unordered_map<Guid, int>::iterator it = state.mCameraIdToGlobalIndex.find(id);
        if (it != state.mCameraIdToGlobalIndex.end()) {
            Component* component = allocators.mCameraAllocator.get(it->second);

            if (component != nullptr) {
                component->deserialize(in);

                return component;
            }
        }
    }
    else if (type == ComponentType<MeshRenderer>::type)
    {
        std::unordered_map<Guid, int>::iterator it = state.mMeshRendererIdToGlobalIndex.find(id);
        if (it != state.mMeshRendererIdToGlobalIndex.end()) {
            Component* component = allocators.mMeshRendererAllocator.get(it->second);

            if (component != nullptr) {
                component->deserialize(in);

                return component;
            }
        }
    }
    else if (type == ComponentType<LineRenderer>::type)
    {
        std::unordered_map<Guid, int>::iterator it = state.mLineRendererIdToGlobalIndex.find(id);
        if (it != state.mLineRendererIdToGlobalIndex.end()) {
            Component* component = allocators.mLineRendererAllocator.get(it->second);

            if (component != nullptr) {
                component->deserialize(in);

                return component;
            }
        }
    }
    else if (type == ComponentType<Light>::type)
    {
        std::unordered_map<Guid, int>::iterator it = state.mLightIdToGlobalIndex.find(id);
        if (it != state.mLightIdToGlobalIndex.end()) {
            Component* component = allocators.mLightAllocator.get(it->second);

            if (component != nullptr) {
                component->deserialize(in);

                return component;
            }
        }
    }
    else if (type == ComponentType<BoxCollider>::type)
    {
        std::unordered_map<Guid, int>::iterator it = state.mBoxColliderIdToGlobalIndex.find(id);
        if (it != state.mBoxColliderIdToGlobalIndex.end()) {
            Component* component = allocators.mBoxColliderAllocator.get(it->second);

            if (component != nullptr) {
                component->deserialize(in);

                return component;
            }
        }
    }
    else if (type == ComponentType<SphereCollider>::type)
    {
        std::unordered_map<Guid, int>::iterator it = state.mSphereColliderIdToGlobalIndex.find(id);
        if (it != state.mSphereColliderIdToGlobalIndex.end()) {
            Component* component = allocators.mSphereColliderAllocator.get(it->second);

            if (component != nullptr) {
                component->deserialize(in);

                return component;
            }
        }
    }
    else if (type == ComponentType<MeshCollider>::type)
    {
        std::unordered_map<Guid, int>::iterator it = state.mMeshColliderIdToGlobalIndex.find(id);
        if (it != state.mMeshColliderIdToGlobalIndex.end()) {
            Component* component = allocators.mMeshColliderAllocator.get(it->second);

            if (component != nullptr) {
                component->deserialize(in);

                return component;
            }
        }
    }
    else if (type == ComponentType<CapsuleCollider>::type)
    {
        std::unordered_map<Guid, int>::iterator it = state.mCapsuleColliderIdToGlobalIndex.find(id);
        if (it != state.mCapsuleColliderIdToGlobalIndex.end()) {
            Component* component = allocators.mCapsuleColliderAllocator.get(it->second);

            if (component != nullptr) {
                component->deserialize(in);

                return component;
            }
        }
    }

    int index = -1;
    Component* component = nullptr;

    if (type == ComponentType<Transform>::type)
    {
        index = (int)allocators.mTransformAllocator.getCount();
        component = allocators.mTransformAllocator.construct(in);

        if (component != nullptr) {
            state.mTransformIdToGlobalIndex[component->getId()] = index;
            state.mIdToGlobalIndex[component->getId()] = index;
            state.mIdToType[component->getId()] = type;
        }
    }
    else if (type == ComponentType<Rigidbody>::type)
    {
        index = (int)allocators.mRigidbodyAllocator.getCount();
        component = allocators.mRigidbodyAllocator.construct(in);

        if (component != nullptr) {
            state.mRigidbodyIdToGlobalIndex[component->getId()] = index;
            state.mIdToGlobalIndex[component->getId()] = index;
            state.mIdToType[component->getId()] = type;
        }
    }
    else if (type == ComponentType<Camera>::type)
    {
        index = (int)allocators.mCameraAllocator.getCount();
        component = allocators.mCameraAllocator.construct(in);

        if (component != nullptr) {
            state.mCameraIdToGlobalIndex[component->getId()] = index;
            state.mIdToGlobalIndex[component->getId()] = index;
            state.mIdToType[component->getId()] = type;
        }
    }
    else if (type == ComponentType<MeshRenderer>::type)
    {
        index = (int)allocators.mMeshRendererAllocator.getCount();
        component = allocators.mMeshRendererAllocator.construct(in);

        if (component != nullptr) {
            state.mMeshRendererIdToGlobalIndex[component->getId()] = index;
            state.mIdToGlobalIndex[component->getId()] = index;
            state.mIdToType[component->getId()] = type;
        }
    }
    else if (type == ComponentType<LineRenderer>::type)
    {
        index = (int)allocators.mLineRendererAllocator.getCount();
        component = allocators.mLineRendererAllocator.construct(in);

        if (component != nullptr) {
            state.mLineRendererIdToGlobalIndex[component->getId()] = index;
            state.mIdToGlobalIndex[component->getId()] = index;
            state.mIdToType[component->getId()] = type;
        }
    }
    else if (type == ComponentType<Light>::type)
    {
        index = (int)allocators.mLightAllocator.getCount();
        component = allocators.mLightAllocator.construct(in);

        if (component != nullptr) {
            state.mLightIdToGlobalIndex[component->getId()] = index;
            state.mIdToGlobalIndex[component->getId()] = index;
            state.mIdToType[component->getId()] = type;
        }
    }
    else if (type == ComponentType<BoxCollider>::type)
    {
        index = (int)allocators.mBoxColliderAllocator.getCount();
        component = allocators.mBoxColliderAllocator.construct(in);

        if (component != nullptr) {
            state.mBoxColliderIdToGlobalIndex[component->getId()] = index;
            state.mIdToGlobalIndex[component->getId()] = index;
            state.mIdToType[component->getId()] = type;
        }
    }
    else if (type == ComponentType<SphereCollider>::type)
    {
        index = (int)allocators.mSphereColliderAllocator.getCount();
        component = allocators.mSphereColliderAllocator.construct(in);

        if (component != nullptr) {
            state.mSphereColliderIdToGlobalIndex[component->getId()] = index;
            state.mIdToGlobalIndex[component->getId()] = index;
            state.mIdToType[component->getId()] = type;
        }
    }
    else if (type == ComponentType<MeshCollider>::type)
    {
        index = (int)allocators.mMeshColliderAllocator.getCount();
        component = allocators.mMeshColliderAllocator.construct(in);

        if (component != nullptr) {
            state.mMeshColliderIdToGlobalIndex[component->getId()] = index;
            state.mIdToGlobalIndex[component->getId()] = index;
            state.mIdToType[component->getId()] = type;
        }
    }
    else if (type == ComponentType<CapsuleCollider>::type)
    {
        index = (int)allocators.mCapsuleColliderAllocator.getCount();
        component = allocators.mCapsuleColliderAllocator.construct(in);

        if (component != nullptr) {
            state.mCapsuleColliderIdToGlobalIndex[component->getId()] = index;
            state.mIdToGlobalIndex[component->getId()] = index;
            state.mIdToType[component->getId()] = type;
        }
    }
    else
    {
        std::string message =
            "Error: Invalid component type (" + std::to_string(type) + ") when trying to load internal component\n";
        Log::error(message.c_str());
        return nullptr;
    }

    return component;
}

System* loadInternalSystem(WorldAllocators& allocators, WorldIdState& state, std::istream& in, const Guid& id, int type)
{
    if (type == SystemType<RenderSystem>::type)
    {
        std::unordered_map<Guid, int>::iterator it = state.mRenderSystemIdToGlobalIndex.find(id);
        if (it != state.mRenderSystemIdToGlobalIndex.end()) {
            System* system = allocators.mRenderSystemAllocator.get(it->second);

            if (system != nullptr) {
                system->deserialize(in);

                return system;
            }
        }
    }
    else if (type == SystemType<PhysicsSystem>::type)
    {
        std::unordered_map<Guid, int>::iterator it = state.mPhysicsSystemIdToGlobalIndex.find(id);
        if (it != state.mPhysicsSystemIdToGlobalIndex.end()) {
            System* system = allocators.mPhysicsSystemAllocator.get(it->second);

            if (system != nullptr) {
                system->deserialize(in);

                return system;
            }
        }
    }
    else if (type == SystemType<CleanUpSystem>::type)
    {
        std::unordered_map<Guid, int>::iterator it = state.mCleanupSystemIdToGlobalIndex.find(id);
        if (it != state.mCleanupSystemIdToGlobalIndex.end()) {
            System* system = allocators.mCleanupSystemAllocator.get(it->second);

            if (system != nullptr) {
                system->deserialize(in);

                return system;
            }
        }
    }
    else if (type == SystemType<DebugSystem>::type)
    {
        std::unordered_map<Guid, int>::iterator it = state.mDebugSystemIdToGlobalIndex.find(id);
        if (it != state.mDebugSystemIdToGlobalIndex.end()) {
            System* system = allocators.mDebugSystemAllocator.get(it->second);

            if (system != nullptr) {
                system->deserialize(in);

                return system;
            }
        }
    }
    else if (type == SystemType<GizmoSystem>::type)
    {
        std::unordered_map<Guid, int>::iterator it = state.mGizmoSystemIdToGlobalIndex.find(id);
        if (it != state.mGizmoSystemIdToGlobalIndex.end()) {
            System* system = allocators.mGizmoSystemAllocator.get(it->second);

            if (system != nullptr) {
                system->deserialize(in);

                return system;
            }
        }
    }

    int index = -1;
    System* system = nullptr;

    if (type == SystemType<RenderSystem>::type)
    {
        index = (int)allocators.mRenderSystemAllocator.getCount();
        system = allocators.mRenderSystemAllocator.construct(in);

        if (system != nullptr) {
            state.mRenderSystemIdToGlobalIndex[system->getId()] = index;
            state.mIdToGlobalIndex[system->getId()] = index;
            state.mIdToType[system->getId()] = type;
        }
    }
    else if (type == SystemType<PhysicsSystem>::type)
    {
        index = (int)allocators.mPhysicsSystemAllocator.getCount();
        system = allocators.mPhysicsSystemAllocator.construct(in);

        if (system != nullptr) {
            state.mPhysicsSystemIdToGlobalIndex[system->getId()] = index;
            state.mIdToGlobalIndex[system->getId()] = index;
            state.mIdToType[system->getId()] = type;
        }
    }
    else if (type == SystemType<CleanUpSystem>::type)
    {
        index = (int)allocators.mCleanupSystemAllocator.getCount();
        system = allocators.mCleanupSystemAllocator.construct(in);

        if (system != nullptr) {
            state.mCleanupSystemIdToGlobalIndex[system->getId()] = index;
            state.mIdToGlobalIndex[system->getId()] = index;
            state.mIdToType[system->getId()] = type;
        }
    }
    else if (type == SystemType<DebugSystem>::type)
    {
        index = (int)allocators.mDebugSystemAllocator.getCount();
        system = allocators.mDebugSystemAllocator.construct(in);

        if (system != nullptr) {
            state.mDebugSystemIdToGlobalIndex[system->getId()] = index;
            state.mIdToGlobalIndex[system->getId()] = index;
            state.mIdToType[system->getId()] = type;
        }
    }
    else if (type == SystemType<GizmoSystem>::type)
    {
        index = (int)allocators.mGizmoSystemAllocator.getCount();
        system = allocators.mGizmoSystemAllocator.construct(in);

        if (system != nullptr) {
            state.mGizmoSystemIdToGlobalIndex[system->getId()] = index;
            state.mIdToGlobalIndex[system->getId()] = index;
            state.mIdToType[system->getId()] = type;
        }
    }
    else
    {
        std::string message =
            "Error: Invalid system type (" + std::to_string(type) + ") when trying to load internal system\n";
        Log::error(message.c_str());
        return nullptr;
    }

    return system;
}

Asset* loadInternalAsset(WorldAllocators& allocators, WorldIdState& state, std::istream& in, const Guid& id, int type)
{
    if (type == AssetType<Shader>::type)
    {
        std::unordered_map<Guid, int>::iterator it = state.mShaderIdToGlobalIndex.find(id);
        if (it != state.mShaderIdToGlobalIndex.end()) {
            Asset* asset = allocators.mShaderAllocator.get(it->second);

            if (asset != nullptr) {
                asset->deserialize(in);

                return asset;
            }
        }
    }
    else if (type == AssetType<Texture2D>::type)
    {
        std::unordered_map<Guid, int>::iterator it = state.mTexture2DIdToGlobalIndex.find(id);
        if (it != state.mTexture2DIdToGlobalIndex.end()) {
            Asset* asset = allocators.mTexture2DAllocator.get(it->second);

            if (asset != nullptr) {
                asset->deserialize(in);

                return asset;
            }
        }
    }
    else if (type == AssetType<Texture3D>::type)
    {
        std::unordered_map<Guid, int>::iterator it = state.mTexture3DIdToGlobalIndex.find(id);
        if (it != state.mTexture3DIdToGlobalIndex.end()) {
            Asset* asset = allocators.mTexture3DAllocator.get(it->second);

            if (asset != nullptr) {
                asset->deserialize(in);

                return asset;
            }
        }
    }
    else if (type == AssetType<Cubemap>::type)
    {
        std::unordered_map<Guid, int>::iterator it = state.mCubemapIdToGlobalIndex.find(id);
        if (it != state.mCubemapIdToGlobalIndex.end()) {
            Asset* asset = allocators.mCubemapAllocator.get(it->second);

            if (asset != nullptr) {
                asset->deserialize(in);

                return asset;
            }
        }
    }
    else if (type == AssetType<Material>::type)
    {
        std::unordered_map<Guid, int>::iterator it = state.mMaterialIdToGlobalIndex.find(id);
        if (it != state.mMaterialIdToGlobalIndex.end()) {
            Asset* asset = allocators.mMaterialAllocator.get(it->second);

            if (asset != nullptr) {
                asset->deserialize(in);

                return asset;
            }
        }
    }
    else if (type == AssetType<Mesh>::type)
    {
        std::unordered_map<Guid, int>::iterator it = state.mMeshIdToGlobalIndex.find(id);
        if (it != state.mMeshIdToGlobalIndex.end()) {
            Asset* asset = allocators.mMeshAllocator.get(it->second);

            if (asset != nullptr) {
                asset->deserialize(in);

                return asset;
            }
        }
    }
    else if (type == AssetType<Font>::type)
    {
        std::unordered_map<Guid, int>::iterator it = state.mFontIdToGlobalIndex.find(id);
        if (it != state.mFontIdToGlobalIndex.end()) {
            Asset* asset = allocators.mFontAllocator.get(it->second);

            if (asset != nullptr) {
                asset->deserialize(in);

                return asset;
            }
        }
    }

    int index = -1;
    Asset* asset = nullptr;

    if (type == AssetType<Shader>::type)
    {
        index = (int)allocators.mShaderAllocator.getCount();
        asset = allocators.mShaderAllocator.construct(in);

        if (asset != nullptr) {
            state.mShaderIdToGlobalIndex[asset->getId()] = index;
            state.mIdToGlobalIndex[asset->getId()] = index;
            state.mIdToType[asset->getId()] = type;
        }
    }
    else if (type == AssetType<Texture2D>::type)
    {
        index = (int)allocators.mTexture2DAllocator.getCount();
        asset = allocators.mTexture2DAllocator.construct(in);

        if (asset != nullptr) {
            state.mTexture2DIdToGlobalIndex[asset->getId()] = index;
            state.mIdToGlobalIndex[asset->getId()] = index;
            state.mIdToType[asset->getId()] = type;
        }
    }
    else if (type == AssetType<Texture3D>::type)
    {
        index = (int)allocators.mTexture3DAllocator.getCount();
        asset = allocators.mTexture3DAllocator.construct(in);

        if (asset != nullptr) {
            state.mTexture3DIdToGlobalIndex[asset->getId()] = index;
            state.mIdToGlobalIndex[asset->getId()] = index;
            state.mIdToType[asset->getId()] = type;
        }
    }
    else if (type == AssetType<Cubemap>::type)
    {
        index = (int)allocators.mCubemapAllocator.getCount();
        asset = allocators.mCubemapAllocator.construct(in);

        if (asset != nullptr) {
            state.mCubemapIdToGlobalIndex[asset->getId()] = index;
            state.mIdToGlobalIndex[asset->getId()] = index;
            state.mIdToType[asset->getId()] = type;
        }
    }
    else if (type == AssetType<Material>::type)
    {
        index = (int)allocators.mMaterialAllocator.getCount();
        asset = allocators.mMaterialAllocator.construct(in);

        if (asset != nullptr) {
            state.mMaterialIdToGlobalIndex[asset->getId()] = index;
            state.mIdToGlobalIndex[asset->getId()] = index;
            state.mIdToType[asset->getId()] = type;
        }
    }
    else if (type == AssetType<Mesh>::type)
    {
        index = (int)allocators.mMeshAllocator.getCount();
        asset = allocators.mMeshAllocator.construct(in);

        if (asset != nullptr) {
            state.mMeshIdToGlobalIndex[asset->getId()] = index;
            state.mIdToGlobalIndex[asset->getId()] = index;
            state.mIdToType[asset->getId()] = type;
        }
    }
    else if (type == AssetType<Font>::type)
    {
        index = (int)allocators.mFontAllocator.getCount();
        asset = allocators.mFontAllocator.construct(in);

        if (asset != nullptr) {
            state.mFontIdToGlobalIndex[asset->getId()] = index;
            state.mIdToGlobalIndex[asset->getId()] = index;
            state.mIdToType[asset->getId()] = type;
        }
    }
    else
    {
        std::string message =
            "Error: Invalid asset type (" + std::to_string(type) + ") when trying to load internal asset\n";
        Log::error(message.c_str());
        return nullptr;
    }

    return asset;
}

Entity *PhysicsEngine::destroyInternalEntity(WorldAllocators& allocators, WorldIdState& state, const Guid& id, int index)
{
    Entity* swap = allocators.mEntityAllocator.destruct(index);

    state.mEntityIdToGlobalIndex.erase(id);
    state.mIdToGlobalIndex.erase(id);
    state.mIdToType.erase(id);

    if (swap != nullptr) {
        state.mEntityIdToGlobalIndex[swap->getId()] = index;
        state.mIdToGlobalIndex[swap->getId()] = index;
        state.mIdToType[swap->getId()] = EntityType<Entity>::type;
    }
    
    return swap;
}

Component *PhysicsEngine::destroyInternalComponent(WorldAllocators& allocators, WorldIdState& state, const Guid& id, int type, int index)
{
    Component* swap = nullptr;

    if (type == ComponentType<Transform>::type)
    {
        swap = allocators.mTransformAllocator.destruct(index);

        state.mTransformIdToGlobalIndex.erase(id);
        state.mIdToGlobalIndex.erase(id);
        state.mIdToType.erase(id);

        if (swap != nullptr) {
            state.mTransformIdToGlobalIndex[swap->getId()] = index;
            state.mIdToGlobalIndex[swap->getId()] = index;
            state.mIdToType[swap->getId()] = type;
        }
    }
    else if (type == ComponentType<Rigidbody>::type)
    {
        swap = allocators.mRigidbodyAllocator.destruct(index);

        state.mRigidbodyIdToGlobalIndex.erase(id);
        state.mIdToGlobalIndex.erase(id);
        state.mIdToType.erase(id);

        if (swap != nullptr) {
            state.mRigidbodyIdToGlobalIndex[swap->getId()] = index;
            state.mIdToGlobalIndex[swap->getId()] = index;
            state.mIdToType[swap->getId()] = type;
        }
    }
    else if (type == ComponentType<Camera>::type)
    {
        swap = allocators.mCameraAllocator.destruct(index);

        state.mCameraIdToGlobalIndex.erase(id);
        state.mIdToGlobalIndex.erase(id);
        state.mIdToType.erase(id);

        if (swap != nullptr) {
            state.mCameraIdToGlobalIndex[swap->getId()] = index;
            state.mIdToGlobalIndex[swap->getId()] = index;
            state.mIdToType[swap->getId()] = type;
        }
    }
    else if (type == ComponentType<MeshRenderer>::type)
    {
        swap = allocators.mMeshRendererAllocator.destruct(index);

        state.mMeshRendererIdToGlobalIndex.erase(id);
        state.mIdToGlobalIndex.erase(id);
        state.mIdToType.erase(id);

        if (swap != nullptr) {
            state.mMeshRendererIdToGlobalIndex[swap->getId()] = index;
            state.mIdToGlobalIndex[swap->getId()] = index;
            state.mIdToType[swap->getId()] = type;
        }
    }
    else if (type == ComponentType<LineRenderer>::type)
    {
        swap = allocators.mLineRendererAllocator.destruct(index);

        state.mLineRendererIdToGlobalIndex.erase(id);
        state.mIdToGlobalIndex.erase(id);
        state.mIdToType.erase(id);

        if (swap != nullptr) {
            state.mLineRendererIdToGlobalIndex[swap->getId()] = index;
            state.mIdToGlobalIndex[swap->getId()] = index;
            state.mIdToType[swap->getId()] = type;
        }
    }
    else if (type == ComponentType<Light>::type)
    {
        swap = allocators.mLightAllocator.destruct(index);

        state.mLightIdToGlobalIndex.erase(id);
        state.mIdToGlobalIndex.erase(id);
        state.mIdToType.erase(id);

        if (swap != nullptr) {
            state.mLightIdToGlobalIndex[swap->getId()] = index;
            state.mIdToGlobalIndex[swap->getId()] = index;
            state.mIdToType[swap->getId()] = type;
        }
    }
    else if (type == ComponentType<BoxCollider>::type)
    {
        swap = allocators.mBoxColliderAllocator.destruct(index);

        state.mBoxColliderIdToGlobalIndex.erase(id);
        state.mIdToGlobalIndex.erase(id);
        state.mIdToType.erase(id);

        if (swap != nullptr) {
            state.mBoxColliderIdToGlobalIndex[swap->getId()] = index;
            state.mIdToGlobalIndex[swap->getId()] = index;
            state.mIdToType[swap->getId()] = type;
        }
    }
    else if (type == ComponentType<SphereCollider>::type)
    {
        swap = allocators.mSphereColliderAllocator.destruct(index);

        state.mSphereColliderIdToGlobalIndex.erase(id);
        state.mIdToGlobalIndex.erase(id);
        state.mIdToType.erase(id);

        if (swap != nullptr) {
            state.mSphereColliderIdToGlobalIndex[swap->getId()] = index;
            state.mIdToGlobalIndex[swap->getId()] = index;
            state.mIdToType[swap->getId()] = type;
        }
    }
    else if (type == ComponentType<MeshCollider>::type)
    {
        swap = allocators.mMeshColliderAllocator.destruct(index);

        state.mMeshColliderIdToGlobalIndex.erase(id);
        state.mIdToGlobalIndex.erase(id);
        state.mIdToType.erase(id);

        if (swap != nullptr) {
            state.mMeshColliderIdToGlobalIndex[swap->getId()] = index;
            state.mIdToGlobalIndex[swap->getId()] = index;
            state.mIdToType[swap->getId()] = type;
        }
    }
    else if (type == ComponentType<CapsuleCollider>::type)
    {
        swap = allocators.mCapsuleColliderAllocator.destruct(index);

        state.mCapsuleColliderIdToGlobalIndex.erase(id);
        state.mIdToGlobalIndex.erase(id);
        state.mIdToType.erase(id);

        if (swap != nullptr) {
            state.mCapsuleColliderIdToGlobalIndex[swap->getId()] = index;
            state.mIdToGlobalIndex[swap->getId()] = index;
            state.mIdToType[swap->getId()] = type;
        }
    }
    else
    {
        std::string message = "Error: Invalid component instance type (" + std::to_string(type) +
                              ") when trying to destroy internal component\n";
        Log::error(message.c_str());
        return nullptr;
    }

    return swap;
}