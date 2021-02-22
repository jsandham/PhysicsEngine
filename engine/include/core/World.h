#ifndef __WORLD_H__
#define __WORLD_H__

#include <assert.h>
#include <string>
#include <unordered_map>

#include "Allocator.h"
#include "Asset.h"
#include "Cubemap.h"
#include "Entity.h"
#include "Font.h"
#include "Guid.h"
#include "Input.h"
#include "LoadInternal.h"
#include "Log.h"
#include "Material.h"
#include "Mesh.h"
#include "PoolAllocator.h"
#include "Serialization.h"
#include "Shader.h"
#include "Texture2D.h"
#include "Texture3D.h"
#include "Util.h"
#include "WorldAllocators.h"
#include "WorldDefaultAssets.h"
#include "WorldIdState.h"
#include "WorldSerialization.h"

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
#include "../systems/GizmoSystem.h"
#include "../systems/PhysicsSystem.h"
#include "../systems/RenderSystem.h"
#include "../systems/System.h"

namespace PhysicsEngine
{
class World
{
  private:
    // allocators for assets, entities, components, and systems
    WorldAllocators mAllocators;

    // id state for assets, entities, components, and systems
    WorldIdState mIdState;

    // default assets
    WorldDefaultAssets mDefaultAssets;

    // all systems in world listed in order they should be updated
    std::vector<System *> mSystems;

    // asset and scene id to filepath
    std::unordered_map<Guid, std::string> mAssetIdToFilepath;
    std::unordered_map<Guid, std::string> mSceneIdToFilepath;

  public:
    World();
    ~World();
    World(const World &other) = delete;
    World &operator=(const World &other) = delete;

    bool loadAsset(const std::string &filePath);
    bool loadScene(const std::string &filePath, bool ignoreSystems = false);
    bool loadSceneFromEditor(const std::string &filePath);

    void latentDestroyEntitiesInWorld();

    size_t getNumberOfEntities() const;
    size_t getNumberOfUpdatingSystems() const;

    template <typename T> size_t getNumberOfSystems() const
    {
        static_assert(std::is_base_of<System, T>(), "'T' is not of type System");

        return getNumberOfSystems_impl<T>(getSystemAllocator_impl<T>());
    }

    template <typename T> size_t getNumberOfComponents() const
    {
        static_assert(std::is_base_of<Component, T>(), "'T' is not of type Component");

        return getNumberOfComponents_impl<T>(getComponentAllocator_impl<T>());
    }

    template <typename T> size_t getNumberOfAssets() const
    {
        static_assert(std::is_base_of<Asset, T>(), "'T' is not of type Asset");

        return getNumberOfAssets_impl(getAssetAllocator_impl<T>());
    }

    template <typename T> T *getSystem()
    {
        static_assert(std::is_base_of<System, T>(), "'T' is not of type System");

        return getSystem_impl(getSystemAllocator_impl<T>());
    }

    template <typename T> T *getComponent(const Guid &entityId)
    {
        static_assert(std::is_base_of<Component, T>(), "'T' is not of type Component");

        return getComponent_impl<T>(getComponentAllocator_impl<T>(), entityId);
    }

    template <typename T> T *addComponent(const Guid &entityId)
    {
        static_assert(std::is_base_of<Component, T>(), "'T' is not of type Component");

        return addComponent_impl<T>(getComponentOrAddAllocator_impl<T>(), entityId);
    }

    template <typename T> T *addComponent(std::istream &in)
    {
        static_assert(std::is_base_of<Component, T>(), "'T' is not of type Component");

        return addComponent_impl<T>(getComponentOrAddAllocator_impl<T>(), in);
    }

    template <typename T> T *addSystem(int order)
    {
        static_assert(std::is_base_of<System, T>(), "'T' is not of type System");

        return addSystem_impl<T>(getSystemOrAddAllocator_impl<T>(), order);
    }

    Entity *getEntityByIndex(int index);
    Entity *getEntityById(const Guid &entityId);
    System *getSystemByUpdateOrder(int order);

    template <typename T> T *getSystemByIndex(int index)
    {
        static_assert(std::is_base_of<System, T>(), "'T' is not of type System");

        return getSystemByIndex_impl(getSystemAllocator_impl<T>(), index);
    }

    template <typename T> T *getSystemById(const Guid &systemId)
    {
        static_assert(std::is_base_of<System, T>(), "'T' is not of type System");

        return getSystemById_impl<T>(getSystemAllocator_impl<T>(), systemId);
    }

    template <typename T> T *getAssetByIndex(int index)
    {
        static_assert(std::is_base_of<Asset, T>(), "'T' is not of type Asset");

        return getAssetByIndex_impl(getAssetAllocator_impl<T>(), index);
    }

    template <typename T> T *getAssetById(const Guid &assetId)
    {
        static_assert(std::is_base_of<Asset, T>(), "'T' is not of type Asset");

        return getAssetById_impl<T>(getAssetAllocator_impl<T>(), assetId);
    }

    template <typename T> T *getComponentByIndex(int index)
    {
        static_assert(std::is_base_of<Component, T>(), "'T' is not of type Component");

        return getComponentByIndex_impl(getComponentAllocator_impl<T>(), index);
    }

    template <typename T> T *getComponentById(const Guid &componentId)
    {
        static_assert(std::is_base_of<Component, T>(), "'T' is not of type Component");

        return getComponentById_impl<T>(getComponentAllocator_impl<T>(), componentId);
    }

    template <typename T> T *createAsset()
    {
        static_assert(std::is_base_of<Asset, T>(), "'T' is not of type Asset");

        return createAsset_impl<T>(getAssetOrAddAllocator_impl<T>());
    }

    template <typename T> T *createAsset(std::istream &in)
    {
        static_assert(std::is_base_of<Asset, T>(), "'T' is not of type Asset");

        return createAsset_impl<T>(getAssetOrAddAllocator_impl<T>(), in);
    }

    int getIndexOf(const Guid &id) const;
    int getTypeOf(const Guid &id) const;

    Entity *createEntity();
    Entity *createEntity(std::istream &in);

    void latentDestroyEntity(const Guid &entityId);
    void immediateDestroyEntity(const Guid &entityId);
    void latentDestroyComponent(const Guid &entityId, const Guid &componentId, int componentType);
    void immediateDestroyComponent(const Guid &entityId, const Guid &componentId, int componentType);
    bool isMarkedForLatentDestroy(const Guid &id);
    void clearIdsMarkedCreatedOrDestroyed();

    std::vector<std::pair<Guid, int>> getComponentsOnEntity(const Guid &entityId);

    std::vector<Guid> getEntityIdsMarkedCreated() const;
    std::vector<Guid> getEntityIdsMarkedLatentDestroy() const;
    std::vector<std::tuple<Guid, Guid, int>> getComponentIdsMarkedCreated() const;
    std::vector<std::tuple<Guid, Guid, int>> getComponentIdsMarkedLatentDestroy() const;

    std::string getAssetFilepath(const Guid &assetId) const;
    std::string getSceneFilepath(const Guid &sceneId) const;

    // Explicit template specializations

    template <> size_t getNumberOfSystems<RenderSystem>() const
    {
        return mAllocators.mRenderSystemAllocator.getCount();
    }

    template <> size_t getNumberOfSystems<PhysicsSystem>() const
    {
        return mAllocators.mPhysicsSystemAllocator.getCount();
    }

    template <> size_t getNumberOfSystems<CleanUpSystem>() const
    {
        return mAllocators.mCleanupSystemAllocator.getCount();
    }

    template <> size_t getNumberOfSystems<DebugSystem>() const
    {
        return mAllocators.mDebugSystemAllocator.getCount();
    }

    template <> size_t getNumberOfSystems<GizmoSystem>() const
    {
        return mAllocators.mGizmoSystemAllocator.getCount();
    }

    template <> size_t getNumberOfComponents<Transform>() const
    {
        return mAllocators.mTransformAllocator.getCount();
    }

    template <> size_t getNumberOfComponents<MeshRenderer>() const
    {
        return mAllocators.mMeshRendererAllocator.getCount();
    }

    template <> size_t getNumberOfComponents<LineRenderer>() const
    {
        return mAllocators.mLineRendererAllocator.getCount();
    }

    template <> size_t getNumberOfComponents<Rigidbody>() const
    {
        return mAllocators.mRigidbodyAllocator.getCount();
    }

    template <> size_t getNumberOfComponents<Camera>() const
    {
        return mAllocators.mCameraAllocator.getCount();
    }

    template <> size_t getNumberOfComponents<Light>() const
    {
        return mAllocators.mLightAllocator.getCount();
    }

    template <> size_t getNumberOfComponents<SphereCollider>() const
    {
        return mAllocators.mSphereColliderAllocator.getCount();
    }

    template <> size_t getNumberOfComponents<BoxCollider>() const
    {
        return mAllocators.mBoxColliderAllocator.getCount();
    }

    template <> size_t getNumberOfComponents<CapsuleCollider>() const
    {
        return mAllocators.mCapsuleColliderAllocator.getCount();
    }

    template <> size_t getNumberOfComponents<MeshCollider>() const
    {
        return mAllocators.mMeshColliderAllocator.getCount();
    }

    template <> size_t getNumberOfAssets<Mesh>() const
    {
        return mAllocators.mMeshAllocator.getCount();
    }

    template <> size_t getNumberOfAssets<Material>() const
    {
        return mAllocators.mMaterialAllocator.getCount();
    }

    template <> size_t getNumberOfAssets<Shader>() const
    {
        return mAllocators.mShaderAllocator.getCount();
    }

    template <> size_t getNumberOfAssets<Texture2D>() const
    {
        return mAllocators.mTexture2DAllocator.getCount();
    }

    template <> size_t getNumberOfAssets<Texture3D>() const
    {
        return mAllocators.mTexture3DAllocator.getCount();
    }

    template <> size_t getNumberOfAssets<Cubemap>() const
    {
        return mAllocators.mCubemapAllocator.getCount();
    }

    template <> size_t getNumberOfAssets<Font>() const
    {
        return mAllocators.mFontAllocator.getCount();
    }

    template <> RenderSystem *getSystem<RenderSystem>()
    {
        return getSystem_impl(&mAllocators.mRenderSystemAllocator);
    }

    template <> PhysicsSystem *getSystem<PhysicsSystem>()
    {
        return getSystem_impl(&mAllocators.mPhysicsSystemAllocator);
    }

    template <> CleanUpSystem *getSystem<CleanUpSystem>()
    {
        return getSystem_impl(&mAllocators.mCleanupSystemAllocator);
    }

    template <> DebugSystem *getSystem<DebugSystem>()
    {
        return getSystem_impl(&mAllocators.mDebugSystemAllocator);
    }

    template <> GizmoSystem *getSystem<GizmoSystem>()
    {
        return getSystem_impl(&mAllocators.mGizmoSystemAllocator);
    }

    template <> Transform *getComponent<Transform>(const Guid &entityId)
    {
        return getComponent_impl(&mAllocators.mTransformAllocator, entityId);
    }

    template <> MeshRenderer *getComponent<MeshRenderer>(const Guid &entityId)
    {
        return getComponent_impl(&mAllocators.mMeshRendererAllocator, entityId);
    }

    template <> LineRenderer *getComponent<LineRenderer>(const Guid &entityId)
    {
        return getComponent_impl(&mAllocators.mLineRendererAllocator, entityId);
    }

    template <> Rigidbody *getComponent<Rigidbody>(const Guid &entityId)
    {
        return getComponent_impl(&mAllocators.mRigidbodyAllocator, entityId);
    }

    template <> Camera *getComponent<Camera>(const Guid &entityId)
    {
        return getComponent_impl(&mAllocators.mCameraAllocator, entityId);
    }

    template <> Light *getComponent<Light>(const Guid &entityId)
    {
        return getComponent_impl(&mAllocators.mLightAllocator, entityId);
    }

    template <> SphereCollider *getComponent<SphereCollider>(const Guid &entityId)
    {
        return getComponent_impl(&mAllocators.mSphereColliderAllocator, entityId);
    }

    template <> BoxCollider *getComponent<BoxCollider>(const Guid &entityId)
    {
        return getComponent_impl(&mAllocators.mBoxColliderAllocator, entityId);
    }

    template <> CapsuleCollider *getComponent<CapsuleCollider>(const Guid &entityId)
    {
        return getComponent_impl(&mAllocators.mCapsuleColliderAllocator, entityId);
    }

    template <> MeshCollider *getComponent<MeshCollider>(const Guid &entityId)
    {
        return getComponent_impl(&mAllocators.mMeshColliderAllocator, entityId);
    }

    template <> Transform *addComponent<Transform>(const Guid &entityId)
    {
        return addComponent_impl(&mAllocators.mTransformAllocator, entityId);
    }

    template <> MeshRenderer *addComponent<MeshRenderer>(const Guid &entityId)
    {
        return addComponent_impl(&mAllocators.mMeshRendererAllocator, entityId);
    }

    template <> LineRenderer *addComponent<LineRenderer>(const Guid &entityId)
    {
        return addComponent_impl(&mAllocators.mLineRendererAllocator, entityId);
    }

    template <> Rigidbody *addComponent<Rigidbody>(const Guid &entityId)
    {
        return addComponent_impl(&mAllocators.mRigidbodyAllocator, entityId);
    }

    template <> Camera *addComponent<Camera>(const Guid &entityId)
    {
        return addComponent_impl(&mAllocators.mCameraAllocator, entityId);
    }

    template <> Light *addComponent<Light>(const Guid &entityId)
    {
        return addComponent_impl(&mAllocators.mLightAllocator, entityId);
    }

    template <> SphereCollider *addComponent<SphereCollider>(const Guid &entityId)
    {
        return addComponent_impl(&mAllocators.mSphereColliderAllocator, entityId);
    }

    template <> BoxCollider *addComponent<BoxCollider>(const Guid &entityId)
    {
        return addComponent_impl(&mAllocators.mBoxColliderAllocator, entityId);
    }

    template <> CapsuleCollider *addComponent<CapsuleCollider>(const Guid &entityId)
    {
        return addComponent_impl(&mAllocators.mCapsuleColliderAllocator, entityId);
    }

    template <> MeshCollider *addComponent<MeshCollider>(const Guid &entityId)
    {
        return addComponent_impl(&mAllocators.mMeshColliderAllocator, entityId);
    }

    template <> Transform *addComponent<Transform>(std::istream &in)
    {
        return addComponent_impl(&mAllocators.mTransformAllocator, in);
    }

    template <> MeshRenderer *addComponent<MeshRenderer>(std::istream &in)
    {
        return addComponent_impl(&mAllocators.mMeshRendererAllocator, in);
    }

    template <> LineRenderer *addComponent<LineRenderer>(std::istream &in)
    {
        return addComponent_impl(&mAllocators.mLineRendererAllocator, in);
    }

    template <> Rigidbody *addComponent<Rigidbody>(std::istream &in)
    {
        return addComponent_impl(&mAllocators.mRigidbodyAllocator, in);
    }

    template <> Camera *addComponent<Camera>(std::istream &in)
    {
        return addComponent_impl(&mAllocators.mCameraAllocator, in);
    }

    template <> Light *addComponent<Light>(std::istream &in)
    {
        return addComponent_impl(&mAllocators.mLightAllocator, in);
    }

    template <> SphereCollider *addComponent<SphereCollider>(std::istream &in)
    {
        return addComponent_impl(&mAllocators.mSphereColliderAllocator, in);
    }

    template <> BoxCollider *addComponent<BoxCollider>(std::istream &in)
    {
        return addComponent_impl(&mAllocators.mBoxColliderAllocator, in);
    }

    template <> CapsuleCollider *addComponent<CapsuleCollider>(std::istream &in)
    {
        return addComponent_impl(&mAllocators.mCapsuleColliderAllocator, in);
    }

    template <> MeshCollider *addComponent<MeshCollider>(std::istream &in)
    {
        return addComponent_impl(&mAllocators.mMeshColliderAllocator, in);
    }

    template <> RenderSystem *addSystem<RenderSystem>(int order)
    {
        return addSystem_impl(&mAllocators.mRenderSystemAllocator, order);
    }

    template <> PhysicsSystem *addSystem<PhysicsSystem>(int order)
    {
        return addSystem_impl(&mAllocators.mPhysicsSystemAllocator, order);
    }

    template <> CleanUpSystem *addSystem<CleanUpSystem>(int order)
    {
        return addSystem_impl(&mAllocators.mCleanupSystemAllocator, order);
    }

    template <> DebugSystem *addSystem<DebugSystem>(int order)
    {
        return addSystem_impl(&mAllocators.mDebugSystemAllocator, order);
    }

    template <> GizmoSystem *addSystem<GizmoSystem>(int order)
    {
        return addSystem_impl(&mAllocators.mGizmoSystemAllocator, order);
    }

    template <> RenderSystem *getSystemByIndex<RenderSystem>(int index)
    {
        return getSystemByIndex_impl(&mAllocators.mRenderSystemAllocator, index);
    }

    template <> PhysicsSystem *getSystemByIndex<PhysicsSystem>(int index)
    {
        return getSystemByIndex_impl(&mAllocators.mPhysicsSystemAllocator, index);
    }

    template <> CleanUpSystem *getSystemByIndex<CleanUpSystem>(int index)
    {
        return getSystemByIndex_impl(&mAllocators.mCleanupSystemAllocator, index);
    }

    template <> DebugSystem *getSystemByIndex<DebugSystem>(int index)
    {
        return getSystemByIndex_impl(&mAllocators.mDebugSystemAllocator, index);
    }

    template <> GizmoSystem *getSystemByIndex<GizmoSystem>(int index)
    {
        return getSystemByIndex_impl(&mAllocators.mGizmoSystemAllocator, index);
    }

    template <> RenderSystem *getSystemById<RenderSystem>(const Guid &systemId)
    {
        return getSystemById_impl(&mAllocators.mRenderSystemAllocator, systemId);
    }

    template <> PhysicsSystem *getSystemById<PhysicsSystem>(const Guid &systemId)
    {
        return getSystemById_impl(&mAllocators.mPhysicsSystemAllocator, systemId);
    }

    template <> CleanUpSystem *getSystemById<CleanUpSystem>(const Guid &systemId)
    {
        return getSystemById_impl(&mAllocators.mCleanupSystemAllocator, systemId);
    }

    template <> DebugSystem *getSystemById<DebugSystem>(const Guid &systemId)
    {
        return getSystemById_impl(&mAllocators.mDebugSystemAllocator, systemId);
    }

    template <> GizmoSystem *getSystemById<GizmoSystem>(const Guid &systemId)
    {
        return getSystemById_impl(&mAllocators.mGizmoSystemAllocator, systemId);
    }

    template <> Mesh *getAssetByIndex<Mesh>(int index)
    {
        return getAssetByIndex_impl(&mAllocators.mMeshAllocator, index);
    }

    template <> Material *getAssetByIndex<Material>(int index)
    {
        return getAssetByIndex_impl(&mAllocators.mMaterialAllocator, index);
    }

    template <> Shader *getAssetByIndex<Shader>(int index)
    {
        return getAssetByIndex_impl(&mAllocators.mShaderAllocator, index);
    }

    template <> Texture2D *getAssetByIndex<Texture2D>(int index)
    {
        return getAssetByIndex_impl(&mAllocators.mTexture2DAllocator, index);
    }

    template <> Texture3D *getAssetByIndex<Texture3D>(int index)
    {
        return getAssetByIndex_impl(&mAllocators.mTexture3DAllocator, index);
    }

    template <> Cubemap *getAssetByIndex<Cubemap>(int index)
    {
        return getAssetByIndex_impl(&mAllocators.mCubemapAllocator, index);
    }

    template <> Font *getAssetByIndex<Font>(int index)
    {
        return getAssetByIndex_impl(&mAllocators.mFontAllocator, index);
    }

    template <> Mesh *getAssetById<Mesh>(const Guid &assetId)
    {
        return getAssetById_impl(&mAllocators.mMeshAllocator, assetId);
    }

    template <> Material *getAssetById<Material>(const Guid &assetId)
    {
        return getAssetById_impl(&mAllocators.mMaterialAllocator, assetId);
    }

    template <> Shader *getAssetById<Shader>(const Guid &assetId)
    {
        return getAssetById_impl(&mAllocators.mShaderAllocator, assetId);
    }

    template <> Texture2D *getAssetById<Texture2D>(const Guid &assetId)
    {
        return getAssetById_impl(&mAllocators.mTexture2DAllocator, assetId);
    }

    template <> Texture3D *getAssetById<Texture3D>(const Guid &assetId)
    {
        return getAssetById_impl(&mAllocators.mTexture3DAllocator, assetId);
    }

    template <> Cubemap *getAssetById<Cubemap>(const Guid &assetId)
    {
        return getAssetById_impl(&mAllocators.mCubemapAllocator, assetId);
    }

    template <> Font *getAssetById<Font>(const Guid &assetId)
    {
        return getAssetById_impl(&mAllocators.mFontAllocator, assetId);
    }

    template <> Transform *getComponentByIndex<Transform>(int index)
    {
        return getComponentByIndex_impl(&mAllocators.mTransformAllocator, index);
    }

    template <> MeshRenderer *getComponentByIndex<MeshRenderer>(int index)
    {
        return getComponentByIndex_impl(&mAllocators.mMeshRendererAllocator, index);
    }

    template <> LineRenderer *getComponentByIndex<LineRenderer>(int index)
    {
        return getComponentByIndex_impl(&mAllocators.mLineRendererAllocator, index);
    }

    template <> Rigidbody *getComponentByIndex<Rigidbody>(int index)
    {
        return getComponentByIndex_impl(&mAllocators.mRigidbodyAllocator, index);
    }

    template <> Camera *getComponentByIndex<Camera>(int index)
    {
        return getComponentByIndex_impl(&mAllocators.mCameraAllocator, index);
    }

    template <> Light *getComponentByIndex<Light>(int index)
    {
        return getComponentByIndex_impl(&mAllocators.mLightAllocator, index);
    }

    template <> SphereCollider *getComponentByIndex<SphereCollider>(int index)
    {
        return getComponentByIndex_impl(&mAllocators.mSphereColliderAllocator, index);
    }

    template <> BoxCollider *getComponentByIndex<BoxCollider>(int index)
    {
        return getComponentByIndex_impl(&mAllocators.mBoxColliderAllocator, index);
    }

    template <> CapsuleCollider *getComponentByIndex<CapsuleCollider>(int index)
    {
        return getComponentByIndex_impl(&mAllocators.mCapsuleColliderAllocator, index);
    }

    template <> MeshCollider *getComponentByIndex<MeshCollider>(int index)
    {
        return getComponentByIndex_impl(&mAllocators.mMeshColliderAllocator, index);
    }

    template <> Transform *getComponentById<Transform>(const Guid &componentId)
    {
        return getComponentById_impl(&mAllocators.mTransformAllocator, componentId);
    }

    template <> MeshRenderer *getComponentById<MeshRenderer>(const Guid &componentId)
    {
        return getComponentById_impl(&mAllocators.mMeshRendererAllocator, componentId);
    }

    template <> LineRenderer *getComponentById<LineRenderer>(const Guid &componentId)
    {
        return getComponentById_impl(&mAllocators.mLineRendererAllocator, componentId);
    }

    template <> Rigidbody *getComponentById<Rigidbody>(const Guid &componentId)
    {
        return getComponentById_impl(&mAllocators.mRigidbodyAllocator, componentId);
    }

    template <> Camera *getComponentById<Camera>(const Guid &componentId)
    {
        return getComponentById_impl(&mAllocators.mCameraAllocator, componentId);
    }

    template <> Light *getComponentById<Light>(const Guid &componentId)
    {
        return getComponentById_impl(&mAllocators.mLightAllocator, componentId);
    }

    template <> SphereCollider *getComponentById<SphereCollider>(const Guid &componentId)
    {
        return getComponentById_impl(&mAllocators.mSphereColliderAllocator, componentId);
    }

    template <> BoxCollider *getComponentById<BoxCollider>(const Guid &componentId)
    {
        return getComponentById_impl(&mAllocators.mBoxColliderAllocator, componentId);
    }

    template <> CapsuleCollider *getComponentById<CapsuleCollider>(const Guid &componentId)
    {
        return getComponentById_impl(&mAllocators.mCapsuleColliderAllocator, componentId);
    }

    template <> MeshCollider *getComponentById<MeshCollider>(const Guid &componentId)
    {
        return getComponentById_impl(&mAllocators.mMeshColliderAllocator, componentId);
    }

    template <> Mesh *createAsset<Mesh>()
    {
        return createAsset_impl(&mAllocators.mMeshAllocator);
    }

    template <> Material *createAsset<Material>()
    {
        return createAsset_impl(&mAllocators.mMaterialAllocator);
    }

    template <> Shader *createAsset<Shader>()
    {
        return createAsset_impl(&mAllocators.mShaderAllocator);
    }

    template <> Texture2D *createAsset<Texture2D>()
    {
        return createAsset_impl(&mAllocators.mTexture2DAllocator);
    }

    template <> Texture3D *createAsset<Texture3D>()
    {
        return createAsset_impl(&mAllocators.mTexture3DAllocator);
    }

    template <> Cubemap *createAsset<Cubemap>()
    {
        return createAsset_impl(&mAllocators.mCubemapAllocator);
    }

    template <> Font *createAsset<Font>()
    {
        return createAsset_impl(&mAllocators.mFontAllocator);
    }

    template <> Mesh *createAsset(std::istream &in)
    {
        return createAsset_impl(&mAllocators.mMeshAllocator, in);
    }

    template <> Material *createAsset(std::istream &in)
    {
        return createAsset_impl(&mAllocators.mMaterialAllocator, in);
    }

    template <> Shader *createAsset(std::istream &in)
    {
        return createAsset_impl(&mAllocators.mShaderAllocator, in);
    }

    template <> Texture2D *createAsset(std::istream &in)
    {
        return createAsset_impl(&mAllocators.mTexture2DAllocator, in);
    }

    template <> Texture3D *createAsset(std::istream &in)
    {
        return createAsset_impl(&mAllocators.mTexture3DAllocator, in);
    }

    template <> Cubemap *createAsset(std::istream &in)
    {
        return createAsset_impl(&mAllocators.mCubemapAllocator, in);
    }

    template <> Font *createAsset(std::istream &in)
    {
        return createAsset_impl(&mAllocators.mFontAllocator, in);
    }

    // default asset getters
    Guid getSphereMesh() const;
    Guid getCubeMesh() const;
    Guid getPlaneMesh() const;
    Guid getColorMaterial() const;
    Guid getSimpleLitMaterial() const;

    Guid getColorLitShaderId() const;
    Guid getNormalLitShaderId() const;
    Guid getTangentLitShaderId() const;
    Guid getFontShaderId() const;
    Guid getGizmoShaderId() const;
    Guid getLineShaderId() const;
    Guid getColorShaderId() const;
    Guid getPositionAndNormalsShaderId() const;
    Guid getSsaoShaderId() const;
    Guid getScreenQuadShaderId() const;
    Guid getNormalMapShaderId() const;
    Guid getDepthMapShaderId() const;
    Guid getShadowDepthMapShaderId() const;
    Guid getShadowDepthCubemapShaderId() const;
    Guid getGbufferShaderId() const;
    Guid getSimpleLitShaderId() const;
    Guid getSimpleLitDeferredShaderId() const;
    Guid getOverdrawShaderId() const;

  private:
    void loadAsset(std::ifstream &in, const ObjectHeader &header);
    void loadEntity(std::ifstream &in, const ObjectHeader &header);
    void loadComponent(std::ifstream &in, const ObjectHeader &header);
    void loadSystem(std::ifstream &in, const ObjectHeader &header);

    template <typename T> size_t getNumberOfSystems_impl(const PoolAllocator<T> *allocator) const
    {
        static_assert(std::is_base_of<System, T>(), "'T' is not of type System");

        return allocator != nullptr ? allocator->getCount() : 0;
    }

    template <typename T> size_t getNumberOfComponents_impl(const PoolAllocator<T> *allocator) const
    {
        static_assert(std::is_base_of<Component, T>(), "'T' is not of type Component");

        return allocator != nullptr ? allocator->getCount() : 0;
    }

    template <typename T> size_t getNumberOfAssets_impl(const PoolAllocator<T> *allocator) const
    {
        static_assert(std::is_base_of<Asset, T>(), "'T' is not of type Asset");

        return allocator != nullptr ? allocator->getCount() : 0;
    }

    template <typename T> T *getSystem_impl(const PoolAllocator<T> *allocator)
    {
        static_assert(std::is_base_of<System, T>(), "'T' is not of type System");

        return allocator != nullptr ? allocator->get(0) : nullptr;
    }

    template <typename T> T *getComponent_impl(const PoolAllocator<T> *allocator, const Guid &entityId)
    {
        static_assert(std::is_base_of<Component, T>(), "'T' is not of type Component");

        if (allocator == nullptr)
        {
            return nullptr;
        }

        auto it1 = mIdState.mEntityIdToComponentIds.find(entityId);

        if (it1 != mIdState.mEntityIdToComponentIds.end())
        {
            std::vector<std::pair<Guid, int>> &componentsOnEntity = it1->second;

            for (size_t i = 0; i < componentsOnEntity.size(); i++)
            {
                if (ComponentType<T>::type == componentsOnEntity[i].second)
                {
                    std::unordered_map<Guid, int>::const_iterator it2 =
                        mIdState.mIdToGlobalIndex.find(componentsOnEntity[i].first);
                    if (it2 != mIdState.mIdToGlobalIndex.end())
                    {
                        return allocator->get(it2->second);
                    }

                    break;
                }
            }
        }

        return nullptr;
    }

    template <typename T> T *addComponent_impl(PoolAllocator<T> *allocator, const Guid &entityId)
    {
        static_assert(std::is_base_of<Component, T>(), "'T' is not of type Component");

        if (getTypeOf(entityId) != EntityType<Entity>::type)
        {
            return nullptr;
        }

        int componentGlobalIndex = (int)allocator->getCount();
        int componentType = ComponentType<T>::type;
        Guid componentId = Guid::newGuid();

        T *component = allocator->construct(componentId);

        if (component != nullptr)
        {
            component->mEntityId = entityId;

            addIdToGlobalIndexMap_impl<T>(componentId, componentGlobalIndex, componentType);

            mIdState.mEntityIdToComponentIds[entityId].push_back(std::make_pair(componentId, componentType));

            mIdState.mComponentIdsMarkedCreated.push_back(std::make_tuple(entityId, componentId, componentType));
        }

        return component;
    }

    template <typename T> T *addComponent_impl(PoolAllocator<T> *allocator, std::istream &in)
    {
        static_assert(std::is_base_of<Component, T>(), "'T' is not of type Component");

        int componentGlobalIndex = (int)allocator->getCount();
        int componentType = ComponentType<T>::type;

        T *component = allocator->construct(in);

        if (component->getId().isInvalid() || component->getEntityId().isInvalid())
        {
            return nullptr;
        }

        addIdToGlobalIndexMap_impl<T>(component->getId(), componentGlobalIndex, componentType);

        mIdState.mEntityIdToComponentIds[component->getEntityId()].push_back(
            std::make_pair(component->getId(), componentType));

        mIdState.mComponentIdsMarkedCreated.push_back(
            std::make_tuple(component->getEntityId(), component->getId(), componentType));

        return component;
    }

    template <typename T> T *addSystem_impl(PoolAllocator<T> *allocator, int order)
    {
        static_assert(std::is_base_of<System, T>(), "'T' is not of type System");

        int systemGlobalIndex = (int)allocator->getCount();
        int systemType = SystemType<T>::type;
        Guid systemId = Guid::newGuid();

        T *system = allocator->construct(systemId);

        if (system != nullptr)
        {
            addIdToGlobalIndexMap_impl<T>(systemId, systemGlobalIndex, systemType);

            size_t locationToInsert = mSystems.size();
            for (size_t i = 0; i < mSystems.size(); i++)
            {
                if (order < mSystems[i]->getOrder())
                {
                    locationToInsert = i;
                    break;
                }
            }

            mSystems.insert(mSystems.begin() + locationToInsert, system);
        }

        return system;
    }

    template <typename T> T *getSystemByIndex_impl(const PoolAllocator<T> *allocator, int index)
    {
        static_assert(std::is_base_of<System, T>(), "'T' is not of type System");

        return allocator != nullptr ? allocator->get(index) : nullptr;
    }

    template <typename T> T *getSystemById_impl(const PoolAllocator<T> *allocator, const Guid &systemId)
    {
        static_assert(std::is_base_of<System, T>(), "'T' is not of type System");

        if (allocator == nullptr || SystemType<T>::type != getTypeOf(systemId))
        {
            return nullptr;
        }

        return getById_impl<T>(mIdState.mIdToGlobalIndex, allocator, systemId);
    }

    template <typename T> T *getAssetByIndex_impl(const PoolAllocator<T> *allocator, int index)
    {
        static_assert(std::is_base_of<Asset, T>(), "'T' is not of type Asset");

        return allocator != nullptr ? allocator->get(index) : nullptr;
    }

    template <typename T> T *getAssetById_impl(const PoolAllocator<T> *allocator, const Guid &assetId)
    {
        static_assert(std::is_base_of<Asset, T>(), "'T' is not of type Asset");

        if (allocator == nullptr || AssetType<T>::type != getTypeOf(assetId))
        {
            return nullptr;
        }

        return getById_impl<T>(mIdState.mIdToGlobalIndex, allocator, assetId);
    }

    template <typename T> T *getComponentByIndex_impl(const PoolAllocator<T> *allocator, int index)
    {
        static_assert(std::is_base_of<Component, T>(), "'T' is not of type Component");

        return allocator != nullptr ? allocator->get(index) : nullptr;
    }

    template <typename T> T *getComponentById_impl(const PoolAllocator<T> *allocator, const Guid &componentId)
    {
        static_assert(std::is_base_of<Component, T>(), "'T' is not of type Component");

        if (allocator == nullptr || ComponentType<T>::type != getTypeOf(componentId))
        {
            return nullptr;
        }

        return getById_impl<T>(mIdState.mIdToGlobalIndex, allocator, componentId);
    }

    template <typename T> T *createAsset_impl(PoolAllocator<T> *allocator)
    {
        static_assert(std::is_base_of<Asset, T>(), "'T' is not of type Asset");

        int index = (int)allocator->getCount();
        int type = AssetType<T>::type;
        Guid id = Guid::newGuid();

        T *asset = allocator->construct(id);

        if (asset != nullptr)
        {
            addIdToGlobalIndexMap_impl<T>(id, index, type);
        }

        return asset;
    }

    template <typename T> T *createAsset_impl(PoolAllocator<T> *allocator, std::istream &in)
    {
        static_assert(std::is_base_of<Asset, T>(), "'T' is not of type Asset");

        int index = (int)allocator->getCount();
        int type = AssetType<T>::type;

        T *asset = allocator->construct(in);

        if (asset != nullptr)
        {
            addIdToGlobalIndexMap_impl<T>(asset->getId(), index, type);
        }

        return asset;
    }

    template <typename T> PoolAllocator<T> *getComponentAllocator_impl()
    {
        static_assert(std::is_base_of<Component, T>(), "'T' is not of type Component");

        std::unordered_map<int, Allocator *>::iterator it = mComponentAllocatorMap.find(ComponentType<T>::type);
        if (it != mComponentAllocatorMap.end())
        {
            return static_cast<PoolAllocator<T> *>(it->second);
        }

        return nullptr;
    }

    template <typename T> PoolAllocator<T> *getComponentOrAddAllocator_impl()
    {
        static_assert(std::is_base_of<Component, T>(), "'T' is not of type Component");

        PoolAllocator<T> *allocator = getComponentAllocator_impl<T>();
        if (allocator == nullptr)
        {
            allocator = new PoolAllocator<T>();
            mComponentAllocatorMap[ComponentType<T>::type] = allocator;
        }

        return allocator;
    }

    template <typename T> PoolAllocator<T> *getSystemAllocator_impl()
    {
        static_assert(std::is_base_of<System, T>(), "'T' is not of type System");

        std::unordered_map<int, Allocator *>::iterator it = mSystemAllocatorMap.find(SystemType<T>::type);
        if (it != mSystemAllocatorMap.end())
        {
            return static_cast<PoolAllocator<T> *>(it->second);
        }

        return nullptr;
    }

    template <typename T> PoolAllocator<T> *getSystemOrAddAllocator_impl()
    {
        static_assert(std::is_base_of<System, T>(), "'T' is not of type System");

        PoolAllocator<T> *allocator = getSystemAllocator_impl<T>();
        if (allocator == nullptr)
        {
            allocator = new PoolAllocator<T>();
            mSystemAllocatorMap[SystemType<T>::type] = allocator;
        }

        return allocator;
    }

    template <typename T> PoolAllocator<T> *getAssetAllocator_impl()
    {
        static_assert(std::is_base_of<Asset, T>(), "'T' is not of type Asset");

        std::unordered_map<int, Allocator *>::iterator it = mAssetAllocatorMap.find(AssetType<T>::type);
        if (it != mAssetAllocatorMap.end())
        {
            return static_cast<PoolAllocator<T> *>(it->second);
        }

        return nullptr;
    }

    template <typename T> PoolAllocator<T> *getAssetOrAddAllocator_impl()
    {
        static_assert(std::is_base_of<Asset, T>(), "'T' is not of type Asset");

        PoolAllocator<T> *allocator = getAssetAllocator_impl<T>();
        if (allocator == nullptr)
        {
            allocator = new PoolAllocator<T>();
            mAssetAllocatorMap[AssetType<T>::type] = allocator;
        }

        return allocator;
    }

    template <typename T>
    T *getById_impl(const std::unordered_map<Guid, int> &idToIndexMap, const PoolAllocator<T> *allocator,
                    const Guid &id)
    {
        std::unordered_map<Guid, int>::const_iterator it = idToIndexMap.find(id);
        if (it != idToIndexMap.end())
        {
            return allocator->get(it->second);
        }
        else
        {
            return nullptr;
        }
    }

    template <typename T> void addIdToGlobalIndexMap_impl(const Guid &id, int index, int type)
    {
        mIdState.mIdToGlobalIndex[id] = index;
        mIdState.mIdToType[id] = type;
    }

    template <int N> void removeIdToGlobalIndexMap_impl(const Guid &id, int type)
    {
        mIdState.mIdToGlobalIndex.erase(id);
        mIdState.mIdToType.erase(id);
    }

    template <typename T> bool verifyData_impl(const std::vector<T> &data) const
    {
        return true;
    }

    // Explicit specializations

    template <>
    Transform *getComponentById_impl<Transform>(const PoolAllocator<Transform> *allocator, const Guid &componentId)
    {
        return getById_impl<Transform>(mIdState.mTransformIdToGlobalIndex, allocator, componentId);
    }

    template <>
    MeshRenderer *getComponentById_impl<MeshRenderer>(const PoolAllocator<MeshRenderer> *allocator,
                                                      const Guid &componentId)
    {
        return getById_impl<MeshRenderer>(mIdState.mMeshRendererIdToGlobalIndex, allocator, componentId);
    }

    template <>
    LineRenderer *getComponentById_impl<LineRenderer>(const PoolAllocator<LineRenderer> *allocator,
                                                      const Guid &componentId)
    {
        return getById_impl<LineRenderer>(mIdState.mLineRendererIdToGlobalIndex, allocator, componentId);
    }

    template <>
    Rigidbody *getComponentById_impl<Rigidbody>(const PoolAllocator<Rigidbody> *allocator, const Guid &componentId)
    {
        return getById_impl<Rigidbody>(mIdState.mRigidbodyIdToGlobalIndex, allocator, componentId);
    }

    template <> Camera *getComponentById_impl<Camera>(const PoolAllocator<Camera> *allocator, const Guid &componentId)
    {
        return getById_impl<Camera>(mIdState.mCameraIdToGlobalIndex, allocator, componentId);
    }

    template <> Light *getComponentById_impl<Light>(const PoolAllocator<Light> *allocator, const Guid &componentId)
    {
        return getById_impl<Light>(mIdState.mLightIdToGlobalIndex, allocator, componentId);
    }

    template <>
    SphereCollider *getComponentById_impl<SphereCollider>(const PoolAllocator<SphereCollider> *allocator,
                                                          const Guid &componentId)
    {
        return getById_impl<SphereCollider>(mIdState.mSphereColliderIdToGlobalIndex, allocator, componentId);
    }

    template <>
    BoxCollider *getComponentById_impl<BoxCollider>(const PoolAllocator<BoxCollider> *allocator,
                                                    const Guid &componentId)
    {
        return getById_impl<BoxCollider>(mIdState.mBoxColliderIdToGlobalIndex, allocator, componentId);
    }

    template <>
    CapsuleCollider *getComponentById_impl<CapsuleCollider>(const PoolAllocator<CapsuleCollider> *allocator,
                                                            const Guid &componentId)
    {
        return getById_impl<CapsuleCollider>(mIdState.mCapsuleColliderIdToGlobalIndex, allocator, componentId);
    }

    template <>
    MeshCollider *getComponentById_impl<MeshCollider>(const PoolAllocator<MeshCollider> *allocator,
                                                      const Guid &componentId)
    {
        return getById_impl<MeshCollider>(mIdState.mMeshColliderIdToGlobalIndex, allocator, componentId);
    }

    template <> Mesh *getAssetById_impl<Mesh>(const PoolAllocator<Mesh> *allocator, const Guid &assetId)
    {
        return getById_impl<Mesh>(mIdState.mMeshIdToGlobalIndex, allocator, assetId);
    }

    template <> Material *getAssetById_impl<Material>(const PoolAllocator<Material> *allocator, const Guid &assetId)
    {
        return getById_impl<Material>(mIdState.mMaterialIdToGlobalIndex, allocator, assetId);
    }

    template <> Shader *getAssetById_impl<Shader>(const PoolAllocator<Shader> *allocator, const Guid &assetId)
    {
        return getById_impl<Shader>(mIdState.mShaderIdToGlobalIndex, allocator, assetId);
    }

    template <> Texture2D *getAssetById_impl<Texture2D>(const PoolAllocator<Texture2D> *allocator, const Guid &assetId)
    {
        return getById_impl<Texture2D>(mIdState.mTexture2DIdToGlobalIndex, allocator, assetId);
    }

    template <> Texture3D *getAssetById_impl<Texture3D>(const PoolAllocator<Texture3D> *allocator, const Guid &assetId)
    {
        return getById_impl<Texture3D>(mIdState.mTexture3DIdToGlobalIndex, allocator, assetId);
    }

    template <> Cubemap *getAssetById_impl<Cubemap>(const PoolAllocator<Cubemap> *allocator, const Guid &assetId)
    {
        return getById_impl<Cubemap>(mIdState.mCubemapIdToGlobalIndex, allocator, assetId);
    }

    template <> Font *getAssetById_impl<Font>(const PoolAllocator<Font> *allocator, const Guid &assetId)
    {
        return getById_impl<Font>(mIdState.mFontIdToGlobalIndex, allocator, assetId);
    }

    template <>
    RenderSystem *getSystemById_impl<RenderSystem>(const PoolAllocator<RenderSystem> *allocator, const Guid &assetId)
    {
        return getById_impl<RenderSystem>(mIdState.mRenderSystemIdToGlobalIndex, allocator, assetId);
    }

    template <>
    PhysicsSystem *getSystemById_impl<PhysicsSystem>(const PoolAllocator<PhysicsSystem> *allocator, const Guid &assetId)
    {
        return getById_impl<PhysicsSystem>(mIdState.mPhysicsSystemIdToGlobalIndex, allocator, assetId);
    }

    template <>
    CleanUpSystem *getSystemById_impl<CleanUpSystem>(const PoolAllocator<CleanUpSystem> *allocator, const Guid &assetId)
    {
        return getById_impl<CleanUpSystem>(mIdState.mCleanupSystemIdToGlobalIndex, allocator, assetId);
    }

    template <>
    DebugSystem *getSystemById_impl<DebugSystem>(const PoolAllocator<DebugSystem> *allocator, const Guid &assetId)
    {
        return getById_impl<DebugSystem>(mIdState.mDebugSystemIdToGlobalIndex, allocator, assetId);
    }
    template <>
    GizmoSystem *getSystemById_impl<GizmoSystem>(const PoolAllocator<GizmoSystem> *allocator, const Guid &assetId)
    {
        return getById_impl<GizmoSystem>(mIdState.mGizmoSystemIdToGlobalIndex, allocator, assetId);
    }

    template <> void addIdToGlobalIndexMap_impl<Entity>(const Guid &id, int index, int type)
    {
        mIdState.mEntityIdToGlobalIndex[id] = index;
        mIdState.mIdToGlobalIndex[id] = index;
        mIdState.mIdToType[id] = type;
    }

    template <> void addIdToGlobalIndexMap_impl<Transform>(const Guid &id, int index, int type)
    {
        mIdState.mTransformIdToGlobalIndex[id] = index;
        mIdState.mIdToGlobalIndex[id] = index;
        mIdState.mIdToType[id] = type;
    }

    template <> void addIdToGlobalIndexMap_impl<MeshRenderer>(const Guid &id, int index, int type)
    {
        mIdState.mMeshRendererIdToGlobalIndex[id] = index;
        mIdState.mIdToGlobalIndex[id] = index;
        mIdState.mIdToType[id] = type;
    }

    template <> void addIdToGlobalIndexMap_impl<LineRenderer>(const Guid &id, int index, int type)
    {
        mIdState.mLineRendererIdToGlobalIndex[id] = index;
        mIdState.mIdToGlobalIndex[id] = index;
        mIdState.mIdToType[id] = type;
    }

    template <> void addIdToGlobalIndexMap_impl<Rigidbody>(const Guid &id, int index, int type)
    {
        mIdState.mRigidbodyIdToGlobalIndex[id] = index;
        mIdState.mIdToGlobalIndex[id] = index;
        mIdState.mIdToType[id] = type;
    }

    template <> void addIdToGlobalIndexMap_impl<Camera>(const Guid &id, int index, int type)
    {
        mIdState.mCameraIdToGlobalIndex[id] = index;
        mIdState.mIdToGlobalIndex[id] = index;
        mIdState.mIdToType[id] = type;
    }

    template <> void addIdToGlobalIndexMap_impl<Light>(const Guid &id, int index, int type)
    {
        mIdState.mLightIdToGlobalIndex[id] = index;
        mIdState.mIdToGlobalIndex[id] = index;
        mIdState.mIdToType[id] = type;
    }

    template <> void addIdToGlobalIndexMap_impl<SphereCollider>(const Guid &id, int index, int type)
    {
        mIdState.mSphereColliderIdToGlobalIndex[id] = index;
        mIdState.mIdToGlobalIndex[id] = index;
        mIdState.mIdToType[id] = type;
    }

    template <> void addIdToGlobalIndexMap_impl<BoxCollider>(const Guid &id, int index, int type)
    {
        mIdState.mBoxColliderIdToGlobalIndex[id] = index;
        mIdState.mIdToGlobalIndex[id] = index;
        mIdState.mIdToType[id] = type;
    }

    template <> void addIdToGlobalIndexMap_impl<CapsuleCollider>(const Guid &id, int index, int type)
    {
        mIdState.mCapsuleColliderIdToGlobalIndex[id] = index;
        mIdState.mIdToGlobalIndex[id] = index;
        mIdState.mIdToType[id] = type;
    }

    template <> void addIdToGlobalIndexMap_impl<MeshCollider>(const Guid &id, int index, int type)
    {
        mIdState.mMeshColliderIdToGlobalIndex[id] = index;
        mIdState.mIdToGlobalIndex[id] = index;
        mIdState.mIdToType[id] = type;
    }

    template <> void addIdToGlobalIndexMap_impl<Mesh>(const Guid &id, int index, int type)
    {
        mIdState.mMeshIdToGlobalIndex[id] = index;
        mIdState.mIdToGlobalIndex[id] = index;
        mIdState.mIdToType[id] = type;
    }

    template <> void addIdToGlobalIndexMap_impl<Material>(const Guid &id, int index, int type)
    {
        mIdState.mMaterialIdToGlobalIndex[id] = index;
        mIdState.mIdToGlobalIndex[id] = index;
        mIdState.mIdToType[id] = type;
    }

    template <> void addIdToGlobalIndexMap_impl<Shader>(const Guid &id, int index, int type)
    {
        mIdState.mShaderIdToGlobalIndex[id] = index;
        mIdState.mIdToGlobalIndex[id] = index;
        mIdState.mIdToType[id] = type;
    }

    template <> void addIdToGlobalIndexMap_impl<Texture2D>(const Guid &id, int index, int type)
    {
        mIdState.mTexture2DIdToGlobalIndex[id] = index;
        mIdState.mIdToGlobalIndex[id] = index;
        mIdState.mIdToType[id] = type;
    }

    template <> void addIdToGlobalIndexMap_impl<Texture3D>(const Guid &id, int index, int type)
    {
        mIdState.mTexture3DIdToGlobalIndex[id] = index;
        mIdState.mIdToGlobalIndex[id] = index;
        mIdState.mIdToType[id] = type;
    }

    template <> void addIdToGlobalIndexMap_impl<Cubemap>(const Guid &id, int index, int type)
    {
        mIdState.mCubemapIdToGlobalIndex[id] = index;
        mIdState.mIdToGlobalIndex[id] = index;
        mIdState.mIdToType[id] = type;
    }

    template <> void addIdToGlobalIndexMap_impl<Font>(const Guid &id, int index, int type)
    {
        mIdState.mFontIdToGlobalIndex[id] = index;
        mIdState.mIdToGlobalIndex[id] = index;
        mIdState.mIdToType[id] = type;
    }

    template <> void addIdToGlobalIndexMap_impl<RenderSystem>(const Guid &id, int index, int type)
    {
        mIdState.mRenderSystemIdToGlobalIndex[id] = index;
        mIdState.mIdToGlobalIndex[id] = index;
        mIdState.mIdToType[id] = type;
    }

    template <> void addIdToGlobalIndexMap_impl<PhysicsSystem>(const Guid &id, int index, int type)
    {
        mIdState.mPhysicsSystemIdToGlobalIndex[id] = index;
        mIdState.mIdToGlobalIndex[id] = index;
        mIdState.mIdToType[id] = type;
    }

    template <> void addIdToGlobalIndexMap_impl<CleanUpSystem>(const Guid &id, int index, int type)
    {
        mIdState.mCleanupSystemIdToGlobalIndex[id] = index;
        mIdState.mIdToGlobalIndex[id] = index;
        mIdState.mIdToType[id] = type;
    }

    template <> void addIdToGlobalIndexMap_impl<DebugSystem>(const Guid &id, int index, int type)
    {
        mIdState.mDebugSystemIdToGlobalIndex[id] = index;
        mIdState.mIdToGlobalIndex[id] = index;
        mIdState.mIdToType[id] = type;
    }

    template <> void addIdToGlobalIndexMap_impl<GizmoSystem>(const Guid &id, int index, int type)
    {
        mIdState.mGizmoSystemIdToGlobalIndex[id] = index;
        mIdState.mIdToGlobalIndex[id] = index;
        mIdState.mIdToType[id] = type;
    }
};
} // namespace PhysicsEngine

#endif