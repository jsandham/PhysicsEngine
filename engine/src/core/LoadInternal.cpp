#include <iostream>

#include "../../include/core/Load.h"
#include "../../include/core/LoadInternal.h"
#include "../../include/core/Log.h"

using namespace PhysicsEngine;

Component *PhysicsEngine::getInternalComponent(const WorldAllocators &allocators, const WorldIdState &state, Id id, int type)
{
    if (type == ComponentType<Transform>::type)
    {
        std::unordered_map<Id, int>::const_iterator it = state.mTransformIdToGlobalIndex.find(id);
        if (it != state.mTransformIdToGlobalIndex.end())
        {
            return allocators.mTransformAllocator.get(it->second);
        }
    }
    else if (type == ComponentType<Rigidbody>::type)
    {
        std::unordered_map<Id, int>::const_iterator it = state.mRigidbodyIdToGlobalIndex.find(id);
        if (it != state.mRigidbodyIdToGlobalIndex.end())
        {
            return allocators.mRigidbodyAllocator.get(it->second);
        }
    }
    else if (type == ComponentType<Camera>::type)
    {
        std::unordered_map<Id, int>::const_iterator it = state.mCameraIdToGlobalIndex.find(id);
        if (it != state.mCameraIdToGlobalIndex.end())
        {
            return allocators.mCameraAllocator.get(it->second);
        }
    }
    else if (type == ComponentType<MeshRenderer>::type)
    {
        std::unordered_map<Id, int>::const_iterator it = state.mMeshRendererIdToGlobalIndex.find(id);
        if (it != state.mMeshRendererIdToGlobalIndex.end())
        {
            return allocators.mMeshRendererAllocator.get(it->second);
        }
    }
    else if (type == ComponentType<SpriteRenderer>::type)
    {
        std::unordered_map<Id, int>::const_iterator it = state.mSpriteRendererIdToGlobalIndex.find(id);
        if (it != state.mSpriteRendererIdToGlobalIndex.end())
        {
            return allocators.mSpriteRendererAllocator.get(it->second);
        }
    }
    else if (type == ComponentType<LineRenderer>::type)
    {
        std::unordered_map<Id, int>::const_iterator it = state.mLineRendererIdToGlobalIndex.find(id);
        if (it != state.mLineRendererIdToGlobalIndex.end())
        {
            return allocators.mLineRendererAllocator.get(it->second);
        }
    }
    else if (type == ComponentType<Light>::type)
    {
        std::unordered_map<Id, int>::const_iterator it = state.mLightIdToGlobalIndex.find(id);
        if (it != state.mLightIdToGlobalIndex.end())
        {
            return allocators.mLightAllocator.get(it->second);
        }
    }
    else if (type == ComponentType<BoxCollider>::type)
    {
        std::unordered_map<Id, int>::const_iterator it = state.mBoxColliderIdToGlobalIndex.find(id);
        if (it != state.mBoxColliderIdToGlobalIndex.end())
        {
            return allocators.mBoxColliderAllocator.get(it->second);
        }
    }
    else if (type == ComponentType<SphereCollider>::type)
    {
        std::unordered_map<Id, int>::const_iterator it = state.mSphereColliderIdToGlobalIndex.find(id);
        if (it != state.mSphereColliderIdToGlobalIndex.end())
        {
            return allocators.mSphereColliderAllocator.get(it->second);
        }
    }
    else if (type == ComponentType<MeshCollider>::type)
    {
        std::unordered_map<Id, int>::const_iterator it = state.mMeshColliderIdToGlobalIndex.find(id);
        if (it != state.mMeshColliderIdToGlobalIndex.end())
        {
            return allocators.mMeshColliderAllocator.get(it->second);
        }
    }
    else if (type == ComponentType<CapsuleCollider>::type)
    {
        std::unordered_map<Id, int>::const_iterator it = state.mCapsuleColliderIdToGlobalIndex.find(id);
        if (it != state.mCapsuleColliderIdToGlobalIndex.end())
        {
            return allocators.mCapsuleColliderAllocator.get(it->second);
        }
    }
    else if (type == ComponentType<Terrain>::type)
    {
        std::unordered_map<Id, int>::const_iterator it = state.mTerrainIdToGlobalIndex.find(id);
        if (it != state.mTerrainIdToGlobalIndex.end())
        {
            return allocators.mTerrainAllocator.get(it->second);
        }
    }

    return nullptr;
}

Asset *PhysicsEngine::getInternalAsset(const WorldAllocators &allocators, const WorldIdState &state, Id id, int type)
{
    if (type == AssetType<Shader>::type)
    {
        std::unordered_map<Id, int>::const_iterator it = state.mShaderIdToGlobalIndex.find(id);
        if (it != state.mShaderIdToGlobalIndex.end())
        {
            return allocators.mShaderAllocator.get(it->second);
        }
    }
    else if (type == AssetType<Mesh>::type)
    {
        std::unordered_map<Id, int>::const_iterator it = state.mMeshIdToGlobalIndex.find(id);
        if (it != state.mMeshIdToGlobalIndex.end())
        {
            return allocators.mMeshAllocator.get(it->second);
        }
    }
    else if (type == AssetType<Material>::type)
    {
        std::unordered_map<Id, int>::const_iterator it = state.mMaterialIdToGlobalIndex.find(id);
        if (it != state.mMaterialIdToGlobalIndex.end())
        {
            return allocators.mMaterialAllocator.get(it->second);
        }
    }
    else if (type == AssetType<Texture2D>::type)
    {
        std::unordered_map<Id, int>::const_iterator it = state.mTexture2DIdToGlobalIndex.find(id);
        if (it != state.mTexture2DIdToGlobalIndex.end())
        {
            return allocators.mTexture2DAllocator.get(it->second);
        }
    }
    else if (type == AssetType<Texture3D>::type)
    {
        std::unordered_map<Id, int>::const_iterator it = state.mTexture3DIdToGlobalIndex.find(id);
        if (it != state.mTexture3DIdToGlobalIndex.end())
        {
            return allocators.mTexture3DAllocator.get(it->second);
        }
    }
    else if (type == AssetType<Cubemap>::type)
    {
        std::unordered_map<Id, int>::const_iterator it = state.mCubemapIdToGlobalIndex.find(id);
        if (it != state.mCubemapIdToGlobalIndex.end())
        {
            return allocators.mCubemapAllocator.get(it->second);
        }
    }
    else if (type == AssetType<RenderTexture>::type)
    {
        std::unordered_map<Id, int>::const_iterator it = state.mRenderTextureIdToGlobalIndex.find(id);
        if (it != state.mRenderTextureIdToGlobalIndex.end())
        {
            return allocators.mRenderTextureAllocator.get(it->second);
        }
    }
    else if (type == AssetType<Font>::type)
    {
        std::unordered_map<Id, int>::const_iterator it = state.mFontIdToGlobalIndex.find(id);
        if (it != state.mFontIdToGlobalIndex.end())
        {
            return allocators.mFontAllocator.get(it->second);
        }
    }

    else if (type == AssetType<Sprite>::type)
    {
        std::unordered_map<Id, int>::const_iterator it = state.mSpriteIdToGlobalIndex.find(id);
        if (it != state.mSpriteIdToGlobalIndex.end())
        {
            return allocators.mSpriteAllocator.get(it->second);
        }
    }

    return nullptr;
}

//Entity *loadEntityFromYAML(World &world, WorldAllocators &allocators, WorldIdState &state, const YAML::Node &in, Id id)
//{
//    return PhysicsEngine::loadInternalEntity(world, allocators, state, in, id);
//}
//
//Component *loadComponentFromYAML(World &world, WorldAllocators &allocators, WorldIdState &state, const YAML::Node &in,
//                                 Id id, int type)
//{
//    if (Component::isInternal(type))
//    {
//        return PhysicsEngine::loadInternalComponent(world, allocators, state, in, id, type);
//    }
//    else
//    {
//        return PhysicsEngine::loadComponent(world, allocators, state, in, id, type);
//    }
//}
//
//System *loadSystemFromYAML(World &world, WorldAllocators &allocators, WorldIdState &state, const YAML::Node &in, Id id, int type)
//{
//    if (System::isInternal(type))
//    {
//        return PhysicsEngine::loadInternalSystem(world, allocators, state, in, id, type);
//    }
//    else
//    {
//        return PhysicsEngine::loadSystem(world, allocators, state, in, id, type);
//    }
//}
//
//Object *loadSceneObjectFromYAML(World &world, WorldAllocators &allocators, WorldIdState &state, const YAML::Node &in)
//{
//    if (in["type"] && in["id"] && in["hide"])
//    {
//        int type = YAML::getValue<int>(in, "type");
//        Guid guid = YAML::getValue<Guid>(in, "id");
//        HideFlag hide = YAML::getValue<HideFlag>(in, "hide");
//
//        if (hide == HideFlag::DontSave)
//        {
//            return nullptr;
//        }
//
//        std::string test = "Loading type: " + std::to_string(type) + " with guid: " + guid.toString() + " \n";
//        Log::info(test.c_str());
//
//        if (PhysicsEngine::isEntity(type))
//        {
//            return loadEntityFromYAML(world, allocators, state, in, id);
//        }
//        else if (PhysicsEngine::isComponent(type))
//        {
//            return loadComponentFromYAML(world, allocators, state, in, id, type);
//        }
//        else if (PhysicsEngine::isSystem(type))
//        {
//            return loadSystemFromYAML(world, allocators, state, in, id, type);
//        }
//    }
//
//    return nullptr;
//}
//
//void loadSceneObjects(World &world, WorldAllocators &allocators, WorldIdState &state, const YAML::Node &in)
//{
//    for (YAML::const_iterator it = in.begin(); it != in.end(); ++it)
//    {
//        if (it->first.IsScalar() && it->second.IsMap())
//        {
//            const Object *object = loadSceneObjectFromYAML(world, allocators, state, it->second);
//
//            if (object == nullptr)
//            {
//                Log::warn("A scene object could not be loaded from scene file. Skipping it.\n");
//            }
//        }
//    }
//}
//
//Scene *loadInternalScene_Impl(World &world, WorldAllocators &allocators, WorldIdState &state, const YAML::Node &in,
//                              Id id)
//{
//    std::unordered_map<Id, int>::iterator it = state.mSceneIdToGlobalIndex.find(id);
//    if (it != state.mSceneIdToGlobalIndex.end())
//    {
//        Scene *scene = allocators.mSceneAllocator.get(it->second);
//
//        if (scene != nullptr)
//        {
//            scene->deserialize(in);
//
//            loadSceneObjects(world, allocators, state, in);
//
//            return scene;
//        }
//    }
//
//    int index = (int)allocators.mSceneAllocator.getCount();
//    Scene *scene = allocators.mSceneAllocator.construct(&world, in);
//
//    if (scene != nullptr)
//    {
//        state.mSceneIdToGlobalIndex[scene->getId()] = index;
//        state.mIdToGlobalIndex[scene->getId()] = index;
//        state.mIdToType[scene->getId()] = SceneType<Scene>::type;
//
//        loadSceneObjects(world, allocators, state, in);
//    }
//
//    return scene;
//}
//
//Entity *loadInternalEntity_Impl(World &world, WorldAllocators &allocators, WorldIdState &state, const YAML::Node &in,
//                                Id id)
//{
//    std::unordered_map<Id, int>::iterator it = state.mEntityIdToGlobalIndex.find(id);
//    if (it != state.mEntityIdToGlobalIndex.end())
//    {
//        Entity *entity = allocators.mEntityAllocator.get(it->second);
//
//        if (entity != nullptr)
//        {
//            entity->deserialize(in);
//
//            return entity;
//        }
//    }
//
//    int index = (int)allocators.mEntityAllocator.getCount();
//    Entity *entity = allocators.mEntityAllocator.construct(&world, in);
//
//    if (entity != nullptr)
//    {
//        state.mEntityIdToGlobalIndex[entity->getId()] = index;
//        state.mIdToGlobalIndex[entity->getId()] = index;
//        state.mIdToType[entity->getId()] = EntityType<Entity>::type;
//    }
//
//    return entity;
//}
//
//Component *loadInternalComponent_Impl(World &world, WorldAllocators &allocators, WorldIdState &state,
//                                      const YAML::Node &in, Id id, int type)
//{
//    if (type == ComponentType<Transform>::type)
//    {
//        std::unordered_map<Id, int>::iterator it = state.mTransformIdToGlobalIndex.find(id);
//        if (it != state.mTransformIdToGlobalIndex.end())
//        {
//            Component *component = allocators.mTransformAllocator.get(it->second);
//
//            if (component != nullptr)
//            {
//                component->deserialize(in);
//
//                return component;
//            }
//        }
//    }
//    else if (type == ComponentType<Rigidbody>::type)
//    {
//        std::unordered_map<Id, int>::iterator it = state.mRigidbodyIdToGlobalIndex.find(id);
//        if (it != state.mRigidbodyIdToGlobalIndex.end())
//        {
//            Component *component = allocators.mRigidbodyAllocator.get(it->second);
//
//            if (component != nullptr)
//            {
//                component->deserialize(in);
//
//                return component;
//            }
//        }
//    }
//    else if (type == ComponentType<Camera>::type)
//    {
//        std::unordered_map<Id, int>::iterator it = state.mCameraIdToGlobalIndex.find(id);
//        if (it != state.mCameraIdToGlobalIndex.end())
//        {
//            Component *component = allocators.mCameraAllocator.get(it->second);
//
//            if (component != nullptr)
//            {
//                component->deserialize(in);
//
//                return component;
//            }
//        }
//    }
//    else if (type == ComponentType<MeshRenderer>::type)
//    {
//        std::unordered_map<Id, int>::iterator it = state.mMeshRendererIdToGlobalIndex.find(id);
//        if (it != state.mMeshRendererIdToGlobalIndex.end())
//        {
//            Component *component = allocators.mMeshRendererAllocator.get(it->second);
//
//            if (component != nullptr)
//            {
//                component->deserialize(in);
//
//                return component;
//            }
//        }
//    }
//    else if (type == ComponentType<SpriteRenderer>::type)
//    {
//        std::unordered_map<Id, int>::iterator it = state.mSpriteRendererIdToGlobalIndex.find(id);
//        if (it != state.mSpriteRendererIdToGlobalIndex.end())
//        {
//            Component *component = allocators.mSpriteRendererAllocator.get(it->second);
//
//            if (component != nullptr)
//            {
//                component->deserialize(in);
//
//                return component;
//            }
//        }
//    }
//    else if (type == ComponentType<LineRenderer>::type)
//    {
//        std::unordered_map<Id, int>::iterator it = state.mLineRendererIdToGlobalIndex.find(id);
//        if (it != state.mLineRendererIdToGlobalIndex.end())
//        {
//            Component *component = allocators.mLineRendererAllocator.get(it->second);
//
//            if (component != nullptr)
//            {
//                component->deserialize(in);
//
//                return component;
//            }
//        }
//    }
//    else if (type == ComponentType<Light>::type)
//    {
//        std::unordered_map<Id, int>::iterator it = state.mLightIdToGlobalIndex.find(id);
//        if (it != state.mLightIdToGlobalIndex.end())
//        {
//            Component *component = allocators.mLightAllocator.get(it->second);
//
//            if (component != nullptr)
//            {
//                component->deserialize(in);
//
//                return component;
//            }
//        }
//    }
//    else if (type == ComponentType<BoxCollider>::type)
//    {
//        std::unordered_map<Id, int>::iterator it = state.mBoxColliderIdToGlobalIndex.find(id);
//        if (it != state.mBoxColliderIdToGlobalIndex.end())
//        {
//            Component *component = allocators.mBoxColliderAllocator.get(it->second);
//
//            if (component != nullptr)
//            {
//                component->deserialize(in);
//
//                return component;
//            }
//        }
//    }
//    else if (type == ComponentType<SphereCollider>::type)
//    {
//        std::unordered_map<Id, int>::iterator it = state.mSphereColliderIdToGlobalIndex.find(id);
//        if (it != state.mSphereColliderIdToGlobalIndex.end())
//        {
//            Component *component = allocators.mSphereColliderAllocator.get(it->second);
//
//            if (component != nullptr)
//            {
//                component->deserialize(in);
//
//                return component;
//            }
//        }
//    }
//    else if (type == ComponentType<MeshCollider>::type)
//    {
//        std::unordered_map<Id, int>::iterator it = state.mMeshColliderIdToGlobalIndex.find(id);
//        if (it != state.mMeshColliderIdToGlobalIndex.end())
//        {
//            Component *component = allocators.mMeshColliderAllocator.get(it->second);
//
//            if (component != nullptr)
//            {
//                component->deserialize(in);
//
//                return component;
//            }
//        }
//    }
//    else if (type == ComponentType<CapsuleCollider>::type)
//    {
//        std::unordered_map<Id, int>::iterator it = state.mCapsuleColliderIdToGlobalIndex.find(id);
//        if (it != state.mCapsuleColliderIdToGlobalIndex.end())
//        {
//            Component *component = allocators.mCapsuleColliderAllocator.get(it->second);
//
//            if (component != nullptr)
//            {
//                component->deserialize(in);
//
//                return component;
//            }
//        }
//    }
//    else if (type == ComponentType<Terrain>::type)
//    {
//        std::unordered_map<Id, int>::iterator it = state.mTerrainIdToGlobalIndex.find(id);
//        if (it != state.mTerrainIdToGlobalIndex.end())
//        {
//            Component *component = allocators.mTerrainAllocator.get(it->second);
//
//            if (component != nullptr)
//            {
//                component->deserialize(in);
//
//                return component;
//            }
//        }
//    }
//
//    int index = -1;
//    Component *component = nullptr;
//
//    if (type == ComponentType<Transform>::type)
//    {
//        index = (int)allocators.mTransformAllocator.getCount();
//        component = allocators.mTransformAllocator.construct(&world, in);
//
//        if (component != nullptr)
//        {
//            state.mTransformIdToGlobalIndex[component->getId()] = index;
//            state.mIdToGlobalIndex[component->getId()] = index;
//            state.mIdToType[component->getId()] = type;
//            state.mEntityIdToComponentIds[component->getEntityId()].push_back(std::make_pair(component->getId(), type));
//        }
//    }
//    else if (type == ComponentType<Rigidbody>::type)
//    {
//        index = (int)allocators.mRigidbodyAllocator.getCount();
//        component = allocators.mRigidbodyAllocator.construct(&world, in);
//
//        if (component != nullptr)
//        {
//            state.mRigidbodyIdToGlobalIndex[component->getId()] = index;
//            state.mIdToGlobalIndex[component->getId()] = index;
//            state.mIdToType[component->getId()] = type;
//            state.mEntityIdToComponentIds[component->getEntityId()].push_back(std::make_pair(component->getId(), type));
//        }
//    }
//    else if (type == ComponentType<Camera>::type)
//    {
//        index = (int)allocators.mCameraAllocator.getCount();
//        component = allocators.mCameraAllocator.construct(&world, in);
//
//        if (component != nullptr)
//        {
//            state.mCameraIdToGlobalIndex[component->getId()] = index;
//            state.mIdToGlobalIndex[component->getId()] = index;
//            state.mIdToType[component->getId()] = type;
//            state.mEntityIdToComponentIds[component->getEntityId()].push_back(std::make_pair(component->getId(), type));
//        }
//    }
//    else if (type == ComponentType<MeshRenderer>::type)
//    {
//        index = (int)allocators.mMeshRendererAllocator.getCount();
//        component = allocators.mMeshRendererAllocator.construct(&world, in);
//
//        if (component != nullptr)
//        {
//            state.mMeshRendererIdToGlobalIndex[component->getId()] = index;
//            state.mIdToGlobalIndex[component->getId()] = index;
//            state.mIdToType[component->getId()] = type;
//            state.mEntityIdToComponentIds[component->getEntityId()].push_back(std::make_pair(component->getId(), type));
//        }
//    }
//    else if (type == ComponentType<SpriteRenderer>::type)
//    {
//        index = (int)allocators.mSpriteRendererAllocator.getCount();
//        component = allocators.mSpriteRendererAllocator.construct(&world, in);
//
//        if (component != nullptr)
//        {
//            state.mSpriteRendererIdToGlobalIndex[component->getId()] = index;
//            state.mIdToGlobalIndex[component->getId()] = index;
//            state.mIdToType[component->getId()] = type;
//            state.mEntityIdToComponentIds[component->getEntityId()].push_back(std::make_pair(component->getId(), type));
//        }
//    }
//    else if (type == ComponentType<LineRenderer>::type)
//    {
//        index = (int)allocators.mLineRendererAllocator.getCount();
//        component = allocators.mLineRendererAllocator.construct(&world, in);
//
//        if (component != nullptr)
//        {
//            state.mLineRendererIdToGlobalIndex[component->getId()] = index;
//            state.mIdToGlobalIndex[component->getId()] = index;
//            state.mIdToType[component->getId()] = type;
//            state.mEntityIdToComponentIds[component->getEntityId()].push_back(std::make_pair(component->getId(), type));
//        }
//    }
//    else if (type == ComponentType<Light>::type)
//    {
//        index = (int)allocators.mLightAllocator.getCount();
//        component = allocators.mLightAllocator.construct(&world, in);
//
//        if (component != nullptr)
//        {
//            state.mLightIdToGlobalIndex[component->getId()] = index;
//            state.mIdToGlobalIndex[component->getId()] = index;
//            state.mIdToType[component->getId()] = type;
//            state.mEntityIdToComponentIds[component->getEntityId()].push_back(std::make_pair(component->getId(), type));
//        }
//    }
//    else if (type == ComponentType<BoxCollider>::type)
//    {
//        index = (int)allocators.mBoxColliderAllocator.getCount();
//        component = allocators.mBoxColliderAllocator.construct(&world, in);
//
//        if (component != nullptr)
//        {
//            state.mBoxColliderIdToGlobalIndex[component->getId()] = index;
//            state.mIdToGlobalIndex[component->getId()] = index;
//            state.mIdToType[component->getId()] = type;
//            state.mEntityIdToComponentIds[component->getEntityId()].push_back(std::make_pair(component->getId(), type));
//        }
//    }
//    else if (type == ComponentType<SphereCollider>::type)
//    {
//        index = (int)allocators.mSphereColliderAllocator.getCount();
//        component = allocators.mSphereColliderAllocator.construct(&world, in);
//
//        if (component != nullptr)
//        {
//            state.mSphereColliderIdToGlobalIndex[component->getId()] = index;
//            state.mIdToGlobalIndex[component->getId()] = index;
//            state.mIdToType[component->getId()] = type;
//            state.mEntityIdToComponentIds[component->getEntityId()].push_back(std::make_pair(component->getId(), type));
//        }
//    }
//    else if (type == ComponentType<MeshCollider>::type)
//    {
//        index = (int)allocators.mMeshColliderAllocator.getCount();
//        component = allocators.mMeshColliderAllocator.construct(&world, in);
//
//        if (component != nullptr)
//        {
//            state.mMeshColliderIdToGlobalIndex[component->getId()] = index;
//            state.mIdToGlobalIndex[component->getId()] = index;
//            state.mIdToType[component->getId()] = type;
//            state.mEntityIdToComponentIds[component->getEntityId()].push_back(std::make_pair(component->getId(), type));
//        }
//    }
//    else if (type == ComponentType<CapsuleCollider>::type)
//    {
//        index = (int)allocators.mCapsuleColliderAllocator.getCount();
//        component = allocators.mCapsuleColliderAllocator.construct(&world, in);
//
//        if (component != nullptr)
//        {
//            state.mCapsuleColliderIdToGlobalIndex[component->getId()] = index;
//            state.mIdToGlobalIndex[component->getId()] = index;
//            state.mIdToType[component->getId()] = type;
//            state.mEntityIdToComponentIds[component->getEntityId()].push_back(std::make_pair(component->getId(), type));
//        }
//    }
//    else if (type == ComponentType<Terrain>::type)
//    {
//        index = (int)allocators.mTerrainAllocator.getCount();
//        component = allocators.mTerrainAllocator.construct(&world, in);
//
//        if (component != nullptr)
//        {
//            state.mTerrainIdToGlobalIndex[component->getId()] = index;
//            state.mIdToGlobalIndex[component->getId()] = index;
//            state.mIdToType[component->getId()] = type;
//            state.mEntityIdToComponentIds[component->getEntityId()].push_back(std::make_pair(component->getId(), type));
//        }
//    }
//    else
//    {
//        std::string message =
//            "Error: Invalid component type (" + std::to_string(type) + ") when trying to load internal component\n";
//        Log::error(message.c_str());
//        return nullptr;
//    }
//
//    return component;
//}
//
//System *loadInternalSystem_Impl(World &world, WorldAllocators &allocators, WorldIdState &state, const YAML::Node &in, Id id,
//                                int type)
//{
//    if (type == SystemType<RenderSystem>::type)
//    {
//        std::unordered_map<Id, int>::iterator it = state.mRenderSystemIdToGlobalIndex.find(id);
//        if (it != state.mRenderSystemIdToGlobalIndex.end())
//        {
//            System *system = allocators.mRenderSystemAllocator.get(it->second);
//
//            if (system != nullptr)
//            {
//                system->deserialize(in);
//
//                return system;
//            }
//        }
//    }
//    else if (type == SystemType<PhysicsSystem>::type)
//    {
//        std::unordered_map<Id, int>::iterator it = state.mPhysicsSystemIdToGlobalIndex.find(id);
//        if (it != state.mPhysicsSystemIdToGlobalIndex.end())
//        {
//            System *system = allocators.mPhysicsSystemAllocator.get(it->second);
//
//            if (system != nullptr)
//            {
//                system->deserialize(in);
//
//                return system;
//            }
//        }
//    }
//    else if (type == SystemType<FreeLookCameraSystem>::type)
//    {
//        std::unordered_map<Id, int>::iterator it = state.mFreeLookCameraSystemIdToGlobalIndex.find(id);
//        if (it != state.mFreeLookCameraSystemIdToGlobalIndex.end())
//        {
//            System *system = allocators.mFreeLookCameraSystemAllocator.get(it->second);
//
//            if (system != nullptr)
//            {
//                system->deserialize(in);
//
//                return system;
//            }
//        }
//    }
//    else if (type == SystemType<CleanUpSystem>::type)
//    {
//        std::unordered_map<Id, int>::iterator it = state.mCleanupSystemIdToGlobalIndex.find(id);
//        if (it != state.mCleanupSystemIdToGlobalIndex.end())
//        {
//            System *system = allocators.mCleanupSystemAllocator.get(it->second);
//
//            if (system != nullptr)
//            {
//                system->deserialize(in);
//
//                return system;
//            }
//        }
//    }
//    else if (type == SystemType<DebugSystem>::type)
//    {
//        std::unordered_map<Id, int>::iterator it = state.mDebugSystemIdToGlobalIndex.find(id);
//        if (it != state.mDebugSystemIdToGlobalIndex.end())
//        {
//            System *system = allocators.mDebugSystemAllocator.get(it->second);
//
//            if (system != nullptr)
//            {
//                system->deserialize(in);
//
//                return system;
//            }
//        }
//    }
//    else if (type == SystemType<GizmoSystem>::type)
//    {
//        std::unordered_map<Id, int>::iterator it = state.mGizmoSystemIdToGlobalIndex.find(id);
//        if (it != state.mGizmoSystemIdToGlobalIndex.end())
//        {
//            System *system = allocators.mGizmoSystemAllocator.get(it->second);
//
//            if (system != nullptr)
//            {
//                system->deserialize(in);
//
//                return system;
//            }
//        }
//    }
//
//    int index = -1;
//    System *system = nullptr;
//
//    if (type == SystemType<RenderSystem>::type)
//    {
//        index = (int)allocators.mRenderSystemAllocator.getCount();
//        system = allocators.mRenderSystemAllocator.construct(&world, in);
//
//        if (system != nullptr)
//        {
//            state.mRenderSystemIdToGlobalIndex[system->getId()] = index;
//            state.mIdToGlobalIndex[system->getId()] = index;
//            state.mIdToType[system->getId()] = type;
//        }
//    }
//    else if (type == SystemType<PhysicsSystem>::type)
//    {
//        index = (int)allocators.mPhysicsSystemAllocator.getCount();
//        system = allocators.mPhysicsSystemAllocator.construct(&world, in);
//
//        if (system != nullptr)
//        {
//            state.mPhysicsSystemIdToGlobalIndex[system->getId()] = index;
//            state.mIdToGlobalIndex[system->getId()] = index;
//            state.mIdToType[system->getId()] = type;
//        }
//    }
//    else if (type == SystemType<FreeLookCameraSystem>::type)
//    {
//        index = (int)allocators.mFreeLookCameraSystemAllocator.getCount();
//        system = allocators.mFreeLookCameraSystemAllocator.construct(&world, in);
//
//        if (system != nullptr)
//        {
//            state.mFreeLookCameraSystemIdToGlobalIndex[system->getId()] = index;
//            state.mIdToGlobalIndex[system->getId()] = index;
//            state.mIdToType[system->getId()] = type;
//        }
//    }
//    else if (type == SystemType<CleanUpSystem>::type)
//    {
//        index = (int)allocators.mCleanupSystemAllocator.getCount();
//        system = allocators.mCleanupSystemAllocator.construct(&world, in);
//
//        if (system != nullptr)
//        {
//            state.mCleanupSystemIdToGlobalIndex[system->getId()] = index;
//            state.mIdToGlobalIndex[system->getId()] = index;
//            state.mIdToType[system->getId()] = type;
//        }
//    }
//    else if (type == SystemType<DebugSystem>::type)
//    {
//        index = (int)allocators.mDebugSystemAllocator.getCount();
//        system = allocators.mDebugSystemAllocator.construct(&world, in);
//
//        if (system != nullptr)
//        {
//            state.mDebugSystemIdToGlobalIndex[system->getId()] = index;
//            state.mIdToGlobalIndex[system->getId()] = index;
//            state.mIdToType[system->getId()] = type;
//        }
//    }
//    else if (type == SystemType<GizmoSystem>::type)
//    {
//        index = (int)allocators.mGizmoSystemAllocator.getCount();
//        system = allocators.mGizmoSystemAllocator.construct(&world, in);
//
//        if (system != nullptr)
//        {
//            state.mGizmoSystemIdToGlobalIndex[system->getId()] = index;
//            state.mIdToGlobalIndex[system->getId()] = index;
//            state.mIdToType[system->getId()] = type;
//        }
//    }
//    else
//    {
//        std::string message =
//            "Error: Invalid system type (" + std::to_string(type) + ") when trying to load internal system\n";
//        Log::error(message.c_str());
//        return nullptr;
//    }
//
//    return system;
//}
//
//Asset *loadInternalAsset_Impl(World &world, WorldAllocators &allocators, WorldIdState &state, const YAML::Node &in,
//                              Id id, int type)
//{
//    if (type == AssetType<Shader>::type)
//    {
//        std::unordered_map<Id, int>::iterator it = state.mShaderIdToGlobalIndex.find(id);
//        if (it != state.mShaderIdToGlobalIndex.end())
//        {
//            Asset *asset = allocators.mShaderAllocator.get(it->second);
//
//            if (asset != nullptr)
//            {
//                asset->deserialize(in);
//
//                return asset;
//            }
//        }
//    }
//    else if (type == AssetType<Texture2D>::type)
//    {
//        std::unordered_map<Id, int>::iterator it = state.mTexture2DIdToGlobalIndex.find(id);
//        if (it != state.mTexture2DIdToGlobalIndex.end())
//        {
//            Asset *asset = allocators.mTexture2DAllocator.get(it->second);
//
//            if (asset != nullptr)
//            {
//                asset->deserialize(in);
//
//                return asset;
//            }
//        }
//    }
//    else if (type == AssetType<Texture3D>::type)
//    {
//        std::unordered_map<Id, int>::iterator it = state.mTexture3DIdToGlobalIndex.find(id);
//        if (it != state.mTexture3DIdToGlobalIndex.end())
//        {
//            Asset *asset = allocators.mTexture3DAllocator.get(it->second);
//
//            if (asset != nullptr)
//            {
//                asset->deserialize(in);
//
//                return asset;
//            }
//        }
//    }
//    else if (type == AssetType<Cubemap>::type)
//    {
//        std::unordered_map<Id, int>::iterator it = state.mCubemapIdToGlobalIndex.find(id);
//        if (it != state.mCubemapIdToGlobalIndex.end())
//        {
//            Asset *asset = allocators.mCubemapAllocator.get(it->second);
//
//            if (asset != nullptr)
//            {
//                asset->deserialize(in);
//
//                return asset;
//            }
//        }
//    }
//    else if (type == AssetType<RenderTexture>::type)
//    {
//        std::unordered_map<Id, int>::iterator it = state.mRenderTextureIdToGlobalIndex.find(id);
//        if (it != state.mRenderTextureIdToGlobalIndex.end())
//        {
//            Asset* asset = allocators.mRenderTextureAllocator.get(it->second);
//
//            if (asset != nullptr)
//            {
//                asset->deserialize(in);
//
//                return asset;
//            }
//        }
//    }
//    else if (type == AssetType<Material>::type)
//    {
//        std::unordered_map<Id, int>::iterator it = state.mMaterialIdToGlobalIndex.find(id);
//        if (it != state.mMaterialIdToGlobalIndex.end())
//        {
//            Asset *asset = allocators.mMaterialAllocator.get(it->second);
//
//            if (asset != nullptr)
//            {
//                asset->deserialize(in);
//
//                return asset;
//            }
//        }
//    }
//    else if (type == AssetType<Mesh>::type)
//    {
//        std::unordered_map<Id, int>::iterator it = state.mMeshIdToGlobalIndex.find(id);
//        if (it != state.mMeshIdToGlobalIndex.end())
//        {
//            Asset *asset = allocators.mMeshAllocator.get(it->second);
//
//            if (asset != nullptr)
//            {
//                asset->deserialize(in);
//
//                return asset;
//            }
//        }
//    }
//    else if (type == AssetType<Font>::type)
//    {
//        std::unordered_map<Id, int>::iterator it = state.mFontIdToGlobalIndex.find(id);
//        if (it != state.mFontIdToGlobalIndex.end())
//        {
//            Asset *asset = allocators.mFontAllocator.get(it->second);
//
//            if (asset != nullptr)
//            {
//                asset->deserialize(in);
//
//                return asset;
//            }
//        }
//    }
//    else if (type == AssetType<Sprite>::type)
//    {
//        std::unordered_map<Id, int>::iterator it = state.mSpriteIdToGlobalIndex.find(id);
//        if (it != state.mSpriteIdToGlobalIndex.end())
//        {
//            Asset *asset = allocators.mSpriteAllocator.get(it->second);
//
//            if (asset != nullptr)
//            {
//                asset->deserialize(in);
//
//                return asset;
//            }
//        }
//    }
//
//    int index = -1;
//    Asset *asset = nullptr;
//
//    if (type == AssetType<Shader>::type)
//    {
//        index = (int)allocators.mShaderAllocator.getCount();
//        asset = allocators.mShaderAllocator.construct(&world, in);
//
//        if (asset != nullptr)
//        {
//            state.mShaderIdToGlobalIndex[asset->getId()] = index;
//            state.mIdToGlobalIndex[asset->getId()] = index;
//            state.mIdToType[asset->getId()] = type;
//        }
//    }
//    else if (type == AssetType<Texture2D>::type)
//    {
//        index = (int)allocators.mTexture2DAllocator.getCount();
//        asset = allocators.mTexture2DAllocator.construct(&world, in);
//
//        if (asset != nullptr)
//        {
//            state.mTexture2DIdToGlobalIndex[asset->getId()] = index;
//            state.mIdToGlobalIndex[asset->getId()] = index;
//            state.mIdToType[asset->getId()] = type;
//        }
//    }
//    else if (type == AssetType<Texture3D>::type)
//    {
//        index = (int)allocators.mTexture3DAllocator.getCount();
//        asset = allocators.mTexture3DAllocator.construct(&world, in);
//
//        if (asset != nullptr)
//        {
//            state.mTexture3DIdToGlobalIndex[asset->getId()] = index;
//            state.mIdToGlobalIndex[asset->getId()] = index;
//            state.mIdToType[asset->getId()] = type;
//        }
//    }
//    else if (type == AssetType<Cubemap>::type)
//    {
//        index = (int)allocators.mCubemapAllocator.getCount();
//        asset = allocators.mCubemapAllocator.construct(&world, in);
//
//        if (asset != nullptr)
//        {
//            state.mCubemapIdToGlobalIndex[asset->getId()] = index;
//            state.mIdToGlobalIndex[asset->getId()] = index;
//            state.mIdToType[asset->getId()] = type;
//        }
//    }
//    else if (type == AssetType<RenderTexture>::type)
//    {
//        index = (int)allocators.mRenderTextureAllocator.getCount();
//        asset = allocators.mRenderTextureAllocator.construct(&world, in);
//
//        if (asset != nullptr)
//        {
//            state.mRenderTextureIdToGlobalIndex[asset->getId()] = index;
//            state.mIdToGlobalIndex[asset->getId()] = index;
//            state.mIdToType[asset->getId()] = type;
//        }
//    }
//    else if (type == AssetType<Material>::type)
//    {
//        index = (int)allocators.mMaterialAllocator.getCount();
//        asset = allocators.mMaterialAllocator.construct(&world, in);
//
//        if (asset != nullptr)
//        {
//            state.mMaterialIdToGlobalIndex[asset->getId()] = index;
//            state.mIdToGlobalIndex[asset->getId()] = index;
//            state.mIdToType[asset->getId()] = type;
//        }
//    }
//    else if (type == AssetType<Mesh>::type)
//    {
//        index = (int)allocators.mMeshAllocator.getCount();
//        asset = allocators.mMeshAllocator.construct(&world, in);
//
//        if (asset != nullptr)
//        {
//            state.mMeshIdToGlobalIndex[asset->getId()] = index;
//            state.mIdToGlobalIndex[asset->getId()] = index;
//            state.mIdToType[asset->getId()] = type;
//        }
//    }
//    else if (type == AssetType<Font>::type)
//    {
//        index = (int)allocators.mFontAllocator.getCount();
//        asset = allocators.mFontAllocator.construct(&world, in);
//
//        if (asset != nullptr)
//        {
//            state.mFontIdToGlobalIndex[asset->getId()] = index;
//            state.mIdToGlobalIndex[asset->getId()] = index;
//            state.mIdToType[asset->getId()] = type;
//        }
//    }
//    else if (type == AssetType<Sprite>::type)
//    {
//        index = (int)allocators.mSpriteAllocator.getCount();
//        asset = allocators.mSpriteAllocator.construct(&world, in);
//
//        if (asset != nullptr)
//        {
//            state.mSpriteIdToGlobalIndex[asset->getId()] = index;
//            state.mIdToGlobalIndex[asset->getId()] = index;
//            state.mIdToType[asset->getId()] = type;
//        }
//    }
//    else
//    {
//        std::string message =
//            "Error: Invalid asset type (" + std::to_string(type) + ") when trying to load internal asset\n";
//        Log::error(message.c_str());
//        return nullptr;
//    }
//
//    return asset;
//}
//
//Scene *PhysicsEngine::loadInternalScene(World &world, WorldAllocators &allocators, WorldIdState &state,
//                                        const YAML::Node &in, Id id)
//{
//    return loadInternalScene_Impl<const YAML::Node>(world, allocators, state, in, id);
//}
//
//Entity *PhysicsEngine::loadInternalEntity(World &world, WorldAllocators &allocators, WorldIdState &state,
//                                          const YAML::Node &in, Id id)
//{
//    return loadInternalEntity_Impl<const YAML::Node>(world, allocators, state, in, id);
//}
//
//Component *PhysicsEngine::loadInternalComponent(World &world, WorldAllocators &allocators, WorldIdState &state,
//                                                const YAML::Node &in, Id id, int type)
//{
//    return loadInternalComponent_Impl<const YAML::Node>(world, allocators, state, in, id, type);
//}
//
//System *PhysicsEngine::loadInternalSystem(World &world, WorldAllocators &allocators, WorldIdState &state,
//                                          const YAML::Node &in, Id id, int type)
//{
//    return loadInternalSystem_Impl<const YAML::Node>(world, allocators, state, in, id, type);
//}
//
//Asset *PhysicsEngine::loadInternalAsset(World &world, WorldAllocators &allocators, WorldIdState &state,
//                                        const YAML::Node &in, Id id, int type)
//{
//    return loadInternalAsset_Impl<const YAML::Node>(world, allocators, state, in, id, type);
//}

Entity *PhysicsEngine::destroyInternalEntity(WorldAllocators &allocators, WorldIdState &state, Id entityId, int index)
{
    Entity *swap = allocators.mEntityAllocator.destruct(index);

    state.mEntityIdToGlobalIndex.erase(entityId);
    state.mIdToGlobalIndex.erase(entityId);
    state.mIdToType.erase(entityId);

    if (swap != nullptr)
    {
        state.mEntityIdToGlobalIndex[swap->getId()] = index;
        state.mIdToGlobalIndex[swap->getId()] = index;
        state.mIdToType[swap->getId()] = EntityType<Entity>::type;
    }

    return swap;
}

Component *PhysicsEngine::destroyInternalComponent(WorldAllocators &allocators, WorldIdState &state,
                                                   Id entityId, Id componentId, int type, int index)
{
    Component *swap = nullptr;

    if (type == ComponentType<Transform>::type)
    {
        swap = allocators.mTransformAllocator.destruct(index);

        state.mTransformIdToGlobalIndex.erase(componentId);
        state.mIdToGlobalIndex.erase(componentId);
        state.mIdToType.erase(componentId);

        if (swap != nullptr)
        {
            state.mTransformIdToGlobalIndex[swap->getId()] = index;
            state.mIdToGlobalIndex[swap->getId()] = index;
            state.mIdToType[swap->getId()] = type;
        }
    }
    else if (type == ComponentType<Rigidbody>::type)
    {
        swap = allocators.mRigidbodyAllocator.destruct(index);

        state.mRigidbodyIdToGlobalIndex.erase(componentId);
        state.mIdToGlobalIndex.erase(componentId);
        state.mIdToType.erase(componentId);

        if (swap != nullptr)
        {
            state.mRigidbodyIdToGlobalIndex[swap->getId()] = index;
            state.mIdToGlobalIndex[swap->getId()] = index;
            state.mIdToType[swap->getId()] = type;
        }
    }
    else if (type == ComponentType<Camera>::type)
    {
        swap = allocators.mCameraAllocator.destruct(index);

        state.mCameraIdToGlobalIndex.erase(componentId);
        state.mIdToGlobalIndex.erase(componentId);
        state.mIdToType.erase(componentId);

        if (swap != nullptr)
        {
            state.mCameraIdToGlobalIndex[swap->getId()] = index;
            state.mIdToGlobalIndex[swap->getId()] = index;
            state.mIdToType[swap->getId()] = type;
        }
    }
    else if (type == ComponentType<MeshRenderer>::type)
    {
        swap = allocators.mMeshRendererAllocator.destruct(index);

        state.mMeshRendererIdToGlobalIndex.erase(componentId);
        state.mIdToGlobalIndex.erase(componentId);
        state.mIdToType.erase(componentId);

        if (swap != nullptr)
        {
            state.mMeshRendererIdToGlobalIndex[swap->getId()] = index;
            state.mIdToGlobalIndex[swap->getId()] = index;
            state.mIdToType[swap->getId()] = type;
        }
    }
    else if (type == ComponentType<SpriteRenderer>::type)
    {
        swap = allocators.mSpriteRendererAllocator.destruct(index);

        state.mSpriteRendererIdToGlobalIndex.erase(componentId);
        state.mIdToGlobalIndex.erase(componentId);
        state.mIdToType.erase(componentId);

        if (swap != nullptr)
        {
            state.mSpriteRendererIdToGlobalIndex[swap->getId()] = index;
            state.mIdToGlobalIndex[swap->getId()] = index;
            state.mIdToType[swap->getId()] = type;
        }
    }
    else if (type == ComponentType<SpriteRenderer>::type)
    {
        swap = allocators.mSpriteRendererAllocator.destruct(index);

        state.mSpriteRendererIdToGlobalIndex.erase(componentId);
        state.mIdToGlobalIndex.erase(componentId);
        state.mIdToType.erase(componentId);

        if (swap != nullptr)
        {
            state.mSpriteRendererIdToGlobalIndex[swap->getId()] = index;
            state.mIdToGlobalIndex[swap->getId()] = index;
            state.mIdToType[swap->getId()] = type;
        }
    }
    else if (type == ComponentType<LineRenderer>::type)
    {
        swap = allocators.mLineRendererAllocator.destruct(index);

        state.mLineRendererIdToGlobalIndex.erase(componentId);
        state.mIdToGlobalIndex.erase(componentId);
        state.mIdToType.erase(componentId);

        if (swap != nullptr)
        {
            state.mLineRendererIdToGlobalIndex[swap->getId()] = index;
            state.mIdToGlobalIndex[swap->getId()] = index;
            state.mIdToType[swap->getId()] = type;
        }
    }
    else if (type == ComponentType<Light>::type)
    {
        swap = allocators.mLightAllocator.destruct(index);

        state.mLightIdToGlobalIndex.erase(componentId);
        state.mIdToGlobalIndex.erase(componentId);
        state.mIdToType.erase(componentId);

        if (swap != nullptr)
        {
            state.mLightIdToGlobalIndex[swap->getId()] = index;
            state.mIdToGlobalIndex[swap->getId()] = index;
            state.mIdToType[swap->getId()] = type;
        }
    }
    else if (type == ComponentType<BoxCollider>::type)
    {
        swap = allocators.mBoxColliderAllocator.destruct(index);

        state.mBoxColliderIdToGlobalIndex.erase(componentId);
        state.mIdToGlobalIndex.erase(componentId);
        state.mIdToType.erase(componentId);

        if (swap != nullptr)
        {
            state.mBoxColliderIdToGlobalIndex[swap->getId()] = index;
            state.mIdToGlobalIndex[swap->getId()] = index;
            state.mIdToType[swap->getId()] = type;
        }
    }
    else if (type == ComponentType<SphereCollider>::type)
    {
        swap = allocators.mSphereColliderAllocator.destruct(index);

        state.mSphereColliderIdToGlobalIndex.erase(componentId);
        state.mIdToGlobalIndex.erase(componentId);
        state.mIdToType.erase(componentId);

        if (swap != nullptr)
        {
            state.mSphereColliderIdToGlobalIndex[swap->getId()] = index;
            state.mIdToGlobalIndex[swap->getId()] = index;
            state.mIdToType[swap->getId()] = type;
        }
    }
    else if (type == ComponentType<MeshCollider>::type)
    {
        swap = allocators.mMeshColliderAllocator.destruct(index);

        state.mMeshColliderIdToGlobalIndex.erase(componentId);
        state.mIdToGlobalIndex.erase(componentId);
        state.mIdToType.erase(componentId);

        if (swap != nullptr)
        {
            state.mMeshColliderIdToGlobalIndex[swap->getId()] = index;
            state.mIdToGlobalIndex[swap->getId()] = index;
            state.mIdToType[swap->getId()] = type;
        }
    }
    else if (type == ComponentType<CapsuleCollider>::type)
    {
        swap = allocators.mCapsuleColliderAllocator.destruct(index);

        state.mCapsuleColliderIdToGlobalIndex.erase(componentId);
        state.mIdToGlobalIndex.erase(componentId);
        state.mIdToType.erase(componentId);

        if (swap != nullptr)
        {
            state.mCapsuleColliderIdToGlobalIndex[swap->getId()] = index;
            state.mIdToGlobalIndex[swap->getId()] = index;
            state.mIdToType[swap->getId()] = type;
        }
    }
    else if (type == ComponentType<Terrain>::type)
    {
        swap = allocators.mTerrainAllocator.destruct(index);

        state.mTerrainIdToGlobalIndex.erase(componentId);
        state.mIdToGlobalIndex.erase(componentId);
        state.mIdToType.erase(componentId);

        if (swap != nullptr)
        {
            state.mTerrainIdToGlobalIndex[swap->getId()] = index;
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

Asset *PhysicsEngine::destroyInternalAsset(WorldAllocators &allocators, WorldIdState &state, Id assetId,
                                           int type, int index)
{
    Asset *swap = nullptr;

    if (type == AssetType<Material>::type)
    {
        swap = allocators.mMaterialAllocator.destruct(index);

        state.mMaterialIdToGlobalIndex.erase(assetId);
        state.mIdToGlobalIndex.erase(assetId);
        state.mIdToType.erase(assetId);

        if (swap != nullptr)
        {
            state.mMaterialIdToGlobalIndex[swap->getId()] = index;
            state.mIdToGlobalIndex[swap->getId()] = index;
            state.mIdToType[swap->getId()] = type;
        }
    }
    else if (type == AssetType<Mesh>::type)
    {
        swap = allocators.mMeshAllocator.destruct(index);

        state.mMeshIdToGlobalIndex.erase(assetId);
        state.mIdToGlobalIndex.erase(assetId);
        state.mIdToType.erase(assetId);

        if (swap != nullptr)
        {
            state.mMeshIdToGlobalIndex[swap->getId()] = index;
            state.mIdToGlobalIndex[swap->getId()] = index;
            state.mIdToType[swap->getId()] = type;
        }
    }
    else if (type == AssetType<Shader>::type)
    {
        swap = allocators.mShaderAllocator.destruct(index);

        state.mShaderIdToGlobalIndex.erase(assetId);
        state.mIdToGlobalIndex.erase(assetId);
        state.mIdToType.erase(assetId);

        if (swap != nullptr)
        {
            state.mShaderIdToGlobalIndex[swap->getId()] = index;
            state.mIdToGlobalIndex[swap->getId()] = index;
            state.mIdToType[swap->getId()] = type;
        }
    }
    else if (type == AssetType<Texture2D>::type)
    {
        swap = allocators.mTexture2DAllocator.destruct(index);

        state.mTexture2DIdToGlobalIndex.erase(assetId);
        state.mIdToGlobalIndex.erase(assetId);
        state.mIdToType.erase(assetId);

        if (swap != nullptr)
        {
            state.mTexture2DIdToGlobalIndex[swap->getId()] = index;
            state.mIdToGlobalIndex[swap->getId()] = index;
            state.mIdToType[swap->getId()] = type;
        }
    }
    else if (type == AssetType<Texture3D>::type)
    {
        swap = allocators.mTexture3DAllocator.destruct(index);

        state.mTexture3DIdToGlobalIndex.erase(assetId);
        state.mIdToGlobalIndex.erase(assetId);
        state.mIdToType.erase(assetId);

        if (swap != nullptr)
        {
            state.mTexture3DIdToGlobalIndex[swap->getId()] = index;
            state.mIdToGlobalIndex[swap->getId()] = index;
            state.mIdToType[swap->getId()] = type;
        }
    }
    else if (type == AssetType<Cubemap>::type)
    {
        swap = allocators.mCubemapAllocator.destruct(index);

        state.mCubemapIdToGlobalIndex.erase(assetId);
        state.mIdToGlobalIndex.erase(assetId);
        state.mIdToType.erase(assetId);

        if (swap != nullptr)
        {
            state.mCubemapIdToGlobalIndex[swap->getId()] = index;
            state.mIdToGlobalIndex[swap->getId()] = index;
            state.mIdToType[swap->getId()] = type;
        }
    }
    else if (type == AssetType<RenderTexture>::type)
    {
        swap = allocators.mRenderTextureAllocator.destruct(index);

        state.mRenderTextureIdToGlobalIndex.erase(assetId);
        state.mIdToGlobalIndex.erase(assetId);
        state.mIdToType.erase(assetId);

        if (swap != nullptr)
        {
            state.mRenderTextureIdToGlobalIndex[swap->getId()] = index;
            state.mIdToGlobalIndex[swap->getId()] = index;
            state.mIdToType[swap->getId()] = type;
        }
    }
    else if (type == AssetType<Font>::type)
    {
        swap = allocators.mFontAllocator.destruct(index);

        state.mFontIdToGlobalIndex.erase(assetId);
        state.mIdToGlobalIndex.erase(assetId);
        state.mIdToType.erase(assetId);

        if (swap != nullptr)
        {
            state.mFontIdToGlobalIndex[swap->getId()] = index;
            state.mIdToGlobalIndex[swap->getId()] = index;
            state.mIdToType[swap->getId()] = type;
        }
    }
    else if (type == AssetType<Sprite>::type)
    {
        swap = allocators.mSpriteAllocator.destruct(index);

        state.mSpriteIdToGlobalIndex.erase(assetId);
        state.mIdToGlobalIndex.erase(assetId);
        state.mIdToType.erase(assetId);

        if (swap != nullptr)
        {
            state.mSpriteIdToGlobalIndex[swap->getId()] = index;
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