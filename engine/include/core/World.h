#ifndef WORLD_H__
#define WORLD_H__

#include <assert.h>
#include <string>
#include <unordered_map>
#include <filesystem>

#define GLM_FORCE_RADIANS

#include "WorldAllocators.h"
#include "WorldIdState.h"

namespace PhysicsEngine
{
class World
{
  private:
    // allocators for assets, entities, components, and systems
    WorldAllocators mAllocators;

    // id state for assets, entities, components, and systems
    WorldIdState mIdState;

    // all systems in world listed in order they should be updated
    std::vector<System *> mSystems;

  public:
    World();
    ~World();
    World(const World &other) = delete;
    World &operator=(const World &other) = delete;

    void loadAssetsInPath(const std::filesystem::path &filePath);
    Asset *loadAssetFromYAML(const std::string &filePath);
    Scene *loadSceneFromYAML(const std::string &filePath);
    bool writeAssetToYAML(const std::string &filePath, const Guid &assetId) const;
    bool writeSceneToYAML(const std::string &filePath, const Guid &sceneId) const;

    void latentDestroyEntitiesInWorld();
    void immediateDestroyEntitiesInWorld();

    size_t getNumberOfScenes() const;
    size_t getNumberOfEntities() const;
    size_t getNumberOfNonHiddenEntities() const;
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

    template <typename T> T *getSystem() const
    {
        static_assert(std::is_base_of<System, T>(), "'T' is not of type System");

        return getSystem_impl(getSystemAllocator_impl<T>());
    }

    template <typename T> T *getComponent(const Guid &entityId) const
    {
        static_assert(std::is_base_of<Component, T>(), "'T' is not of type Component");

        return getComponent_impl<T>(getComponentAllocator_impl<T>(), entityId);
    }

    template <typename T> T *addComponent(const Guid &entityId)
    {
        static_assert(std::is_base_of<Component, T>(), "'T' is not of type Component");

        return addComponent_impl<T>(getComponentOrAddAllocator_impl<T>(), entityId);
    }

    template <typename T> T *addSystem(size_t order)
    {
        static_assert(std::is_base_of<System, T>(), "'T' is not of type System");

        return addSystem_impl<T>(getSystemOrAddAllocator_impl<T>(), order);
    }

    Scene *getSceneByIndex(size_t index) const;
    Scene *getSceneById(const Guid &sceneId) const;
    Entity *getEntityByIndex(size_t index) const;
    Entity *getEntityById(const Guid &entityId) const;
    System *getSystemByUpdateOrder(size_t order) const;

    template <typename T> T *getSystemByIndex(size_t index) const
    {
        static_assert(std::is_base_of<System, T>(), "'T' is not of type System");

        return getSystemByIndex_impl(getSystemAllocator_impl<T>(), index);
    }

    template <typename T> T *getSystemById(const Guid &systemId) const
    {
        static_assert(std::is_base_of<System, T>(), "'T' is not of type System");

        return getSystemById_impl<T>(getSystemAllocator_impl<T>(), systemId);
    }

    template <typename T> T *getAssetByIndex(size_t index) const
    {
        static_assert(std::is_base_of<Asset, T>(), "'T' is not of type Asset");

        return getAssetByIndex_impl(getAssetAllocator_impl<T>(), index);
    }

    template <typename T> T *getAssetById(const Guid &assetId) const
    {
        static_assert(std::is_base_of<Asset, T>(), "'T' is not of type Asset");

        return getAssetById_impl<T>(getAssetAllocator_impl<T>(), assetId);
    }

    template <typename T> T *getComponentByIndex(size_t index) const
    {
        static_assert(std::is_base_of<Component, T>(), "'T' is not of type Component");

        return getComponentByIndex_impl(getComponentAllocator_impl<T>(), index);
    }

    template <typename T> T *getComponentById(const Guid &componentId) const
    {
        static_assert(std::is_base_of<Component, T>(), "'T' is not of type Component");

        return getComponentById_impl<T>(getComponentAllocator_impl<T>(), componentId);
    }

    template <typename T> T *createAsset()
    {
        static_assert(std::is_base_of<Asset, T>(), "'T' is not of type Asset");

        return createAsset_impl<T>(getAssetOrAddAllocator_impl<T>());
    }

    int getIndexOf(const Guid &id) const;
    int getTypeOf(const Guid &id) const;

    Scene *createScene();
    Entity *createEntity();

    void latentDestroyEntity(const Guid &entityId);
    void immediateDestroyEntity(const Guid &entityId);
    void latentDestroyComponent(const Guid &entityId, const Guid &componentId, int componentType);
    void immediateDestroyComponent(const Guid &entityId, const Guid &componentId, int componentType);
    void latentDestroyAsset(const Guid &assetId, int assetType);
    void immediateDestroyAsset(const Guid &assetId, int assetType);

    bool isMarkedForLatentDestroy(const Guid &id);
    void clearIdsMarkedCreatedOrDestroyed();

    std::vector<std::pair<Guid, int>> getComponentsOnEntity(const Guid &entityId) const;

    std::vector<Guid> getEntityIdsMarkedCreated() const;
    std::vector<Guid> getEntityIdsMarkedLatentDestroy() const;
    std::vector<std::tuple<Guid, Guid, int>> getComponentIdsMarkedCreated() const;
    std::vector<std::tuple<Guid, Guid, int>> getComponentIdsMarkedLatentDestroy() const;

    std::string getAssetFilepath(const Guid &assetId) const;
    std::string getSceneFilepath(const Guid &sceneId) const;

    Guid getAssetId(const std::string& filepath) const;
    Guid getSceneId(const std::string& filepath) const;

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

    template <> size_t getNumberOfSystems<FreeLookCameraSystem>() const
    {
        return mAllocators.mFreeLookCameraSystemAllocator.getCount();
    }

    template <> size_t getNumberOfComponents<Transform>() const
    {
        return mAllocators.mTransformAllocator.getCount();
    }

    template <> size_t getNumberOfComponents<MeshRenderer>() const
    {
        return mAllocators.mMeshRendererAllocator.getCount();
    }

    template <> size_t getNumberOfComponents<SpriteRenderer>() const
    {
        return mAllocators.mSpriteRendererAllocator.getCount();
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

    template <> size_t getNumberOfAssets<RenderTexture>() const
    {
        return mAllocators.mRenderTextureAllocator.getCount();
    }

    template <> size_t getNumberOfAssets<Font>() const
    {
        return mAllocators.mFontAllocator.getCount();
    }

    template <> size_t getNumberOfAssets<Sprite>() const
    {
        return mAllocators.mSpriteAllocator.getCount();
    }

    template <> RenderSystem *getSystem<RenderSystem>() const
    {
        return getSystem_impl(&mAllocators.mRenderSystemAllocator);
    }

    template <> PhysicsSystem *getSystem<PhysicsSystem>() const
    {
        return getSystem_impl(&mAllocators.mPhysicsSystemAllocator);
    }

    template <> CleanUpSystem *getSystem<CleanUpSystem>() const
    {
        return getSystem_impl(&mAllocators.mCleanupSystemAllocator);
    }

    template <> DebugSystem *getSystem<DebugSystem>() const
    {
        return getSystem_impl(&mAllocators.mDebugSystemAllocator);
    }

    template <> GizmoSystem *getSystem<GizmoSystem>() const
    {
        return getSystem_impl(&mAllocators.mGizmoSystemAllocator);
    }

    template <> FreeLookCameraSystem* getSystem<FreeLookCameraSystem>() const
    {
        return getSystem_impl(&mAllocators.mFreeLookCameraSystemAllocator);
    }

    template <> Transform *getComponent<Transform>(const Guid &entityId) const
    {
        return getComponent_impl(&mAllocators.mTransformAllocator, entityId);
    }

    template <> MeshRenderer *getComponent<MeshRenderer>(const Guid &entityId) const
    {
        return getComponent_impl(&mAllocators.mMeshRendererAllocator, entityId);
    }

    template <> SpriteRenderer *getComponent<SpriteRenderer>(const Guid &entityId) const
    {
        return getComponent_impl(&mAllocators.mSpriteRendererAllocator, entityId);
    }

    template <> LineRenderer *getComponent<LineRenderer>(const Guid &entityId) const
    {
        return getComponent_impl(&mAllocators.mLineRendererAllocator, entityId);
    }

    template <> Rigidbody *getComponent<Rigidbody>(const Guid &entityId) const
    {
        return getComponent_impl(&mAllocators.mRigidbodyAllocator, entityId);
    }

    template <> Camera *getComponent<Camera>(const Guid &entityId) const
    {
        return getComponent_impl(&mAllocators.mCameraAllocator, entityId);
    }

    template <> Light *getComponent<Light>(const Guid &entityId) const
    {
        return getComponent_impl(&mAllocators.mLightAllocator, entityId);
    }

    template <> SphereCollider *getComponent<SphereCollider>(const Guid &entityId) const
    {
        return getComponent_impl(&mAllocators.mSphereColliderAllocator, entityId);
    }

    template <> BoxCollider *getComponent<BoxCollider>(const Guid &entityId) const
    {
        return getComponent_impl(&mAllocators.mBoxColliderAllocator, entityId);
    }

    template <> CapsuleCollider *getComponent<CapsuleCollider>(const Guid &entityId) const
    {
        return getComponent_impl(&mAllocators.mCapsuleColliderAllocator, entityId);
    }

    template <> MeshCollider *getComponent<MeshCollider>(const Guid &entityId) const
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

    template <> SpriteRenderer *addComponent<SpriteRenderer>(const Guid &entityId)
    {
        return addComponent_impl(&mAllocators.mSpriteRendererAllocator, entityId);
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

    template <> RenderSystem *addSystem<RenderSystem>(size_t order)
    {
        return addSystem_impl(&mAllocators.mRenderSystemAllocator, order);
    }

    template <> PhysicsSystem *addSystem<PhysicsSystem>(size_t order)
    {
        return addSystem_impl(&mAllocators.mPhysicsSystemAllocator, order);
    }

    template <> CleanUpSystem *addSystem<CleanUpSystem>(size_t order)
    {
        return addSystem_impl(&mAllocators.mCleanupSystemAllocator, order);
    }

    template <> DebugSystem *addSystem<DebugSystem>(size_t order)
    {
        return addSystem_impl(&mAllocators.mDebugSystemAllocator, order);
    }

    template <> GizmoSystem *addSystem<GizmoSystem>(size_t order)
    {
        return addSystem_impl(&mAllocators.mGizmoSystemAllocator, order);
    }

    template <> FreeLookCameraSystem* addSystem<FreeLookCameraSystem>(size_t order)
    {
        return addSystem_impl(&mAllocators.mFreeLookCameraSystemAllocator, order);
    }

    template <> RenderSystem *getSystemByIndex<RenderSystem>(size_t index) const
    {
        return getSystemByIndex_impl(&mAllocators.mRenderSystemAllocator, index);
    }

    template <> PhysicsSystem *getSystemByIndex<PhysicsSystem>(size_t index) const
    {
        return getSystemByIndex_impl(&mAllocators.mPhysicsSystemAllocator, index);
    }

    template <> CleanUpSystem *getSystemByIndex<CleanUpSystem>(size_t index) const
    {
        return getSystemByIndex_impl(&mAllocators.mCleanupSystemAllocator, index);
    }

    template <> DebugSystem *getSystemByIndex<DebugSystem>(size_t index) const
    {
        return getSystemByIndex_impl(&mAllocators.mDebugSystemAllocator, index);
    }

    template <> GizmoSystem *getSystemByIndex<GizmoSystem>(size_t index) const
    {
        return getSystemByIndex_impl(&mAllocators.mGizmoSystemAllocator, index);
    }

    template <> FreeLookCameraSystem* getSystemByIndex<FreeLookCameraSystem>(size_t index) const
    {
        return getSystemByIndex_impl(&mAllocators.mFreeLookCameraSystemAllocator, index);
    }

    template <> RenderSystem *getSystemById<RenderSystem>(const Guid &systemId) const
    {
        return getSystemById_impl(&mAllocators.mRenderSystemAllocator, systemId);
    }

    template <> PhysicsSystem *getSystemById<PhysicsSystem>(const Guid &systemId) const
    {
        return getSystemById_impl(&mAllocators.mPhysicsSystemAllocator, systemId);
    }

    template <> CleanUpSystem *getSystemById<CleanUpSystem>(const Guid &systemId) const
    {
        return getSystemById_impl(&mAllocators.mCleanupSystemAllocator, systemId);
    }

    template <> DebugSystem *getSystemById<DebugSystem>(const Guid &systemId) const
    {
        return getSystemById_impl(&mAllocators.mDebugSystemAllocator, systemId);
    }

    template <> GizmoSystem *getSystemById<GizmoSystem>(const Guid &systemId) const
    {
        return getSystemById_impl(&mAllocators.mGizmoSystemAllocator, systemId);
    }

    template <> FreeLookCameraSystem* getSystemById<FreeLookCameraSystem>(const Guid& systemId) const
    {
        return getSystemById_impl(&mAllocators.mFreeLookCameraSystemAllocator, systemId);
    }

    template <> Mesh *getAssetByIndex<Mesh>(size_t index) const
    {
        return getAssetByIndex_impl(&mAllocators.mMeshAllocator, index);
    }

    template <> Material *getAssetByIndex<Material>(size_t index) const
    {
        return getAssetByIndex_impl(&mAllocators.mMaterialAllocator, index);
    }

    template <> Shader *getAssetByIndex<Shader>(size_t index) const
    {
        return getAssetByIndex_impl(&mAllocators.mShaderAllocator, index);
    }

    template <> Texture2D *getAssetByIndex<Texture2D>(size_t index) const
    {
        return getAssetByIndex_impl(&mAllocators.mTexture2DAllocator, index);
    }

    template <> Texture3D *getAssetByIndex<Texture3D>(size_t index) const
    {
        return getAssetByIndex_impl(&mAllocators.mTexture3DAllocator, index);
    }

    template <> Cubemap *getAssetByIndex<Cubemap>(size_t index) const
    {
        return getAssetByIndex_impl(&mAllocators.mCubemapAllocator, index);
    }

    template <> RenderTexture* getAssetByIndex<RenderTexture>(size_t index) const
    {
        return getAssetByIndex_impl(&mAllocators.mRenderTextureAllocator, index);
    }

    template <> Font *getAssetByIndex<Font>(size_t index) const
    {
        return getAssetByIndex_impl(&mAllocators.mFontAllocator, index);
    }

    template <> Sprite *getAssetByIndex<Sprite>(size_t index) const
    {
        return getAssetByIndex_impl(&mAllocators.mSpriteAllocator, index);
    }

    template <> Mesh *getAssetById<Mesh>(const Guid &assetId) const
    {
        return getAssetById_impl(&mAllocators.mMeshAllocator, assetId);
    }

    template <> Material *getAssetById<Material>(const Guid &assetId) const
    {
        return getAssetById_impl(&mAllocators.mMaterialAllocator, assetId);
    }

    template <> Shader *getAssetById<Shader>(const Guid &assetId) const
    {
        return getAssetById_impl(&mAllocators.mShaderAllocator, assetId);
    }

    template <> Texture2D *getAssetById<Texture2D>(const Guid &assetId) const
    {
        return getAssetById_impl(&mAllocators.mTexture2DAllocator, assetId);
    }

    template <> Texture3D *getAssetById<Texture3D>(const Guid &assetId) const
    {
        return getAssetById_impl(&mAllocators.mTexture3DAllocator, assetId);
    }

    template <> Cubemap *getAssetById<Cubemap>(const Guid &assetId) const
    {
        return getAssetById_impl(&mAllocators.mCubemapAllocator, assetId);
    }

    template <> RenderTexture* getAssetById<RenderTexture>(const Guid& assetId) const
    {
        return getAssetById_impl(&mAllocators.mRenderTextureAllocator, assetId);
    }

    template <> Font *getAssetById<Font>(const Guid &assetId) const
    {
        return getAssetById_impl(&mAllocators.mFontAllocator, assetId);
    }

    template <> Sprite *getAssetById<Sprite>(const Guid &assetId) const
    {
        return getAssetById_impl(&mAllocators.mSpriteAllocator, assetId);
    }

    template <> Transform *getComponentByIndex<Transform>(size_t index) const
    {
        return getComponentByIndex_impl(&mAllocators.mTransformAllocator, index);
    }

    template <> MeshRenderer *getComponentByIndex<MeshRenderer>(size_t index) const
    {
        return getComponentByIndex_impl(&mAllocators.mMeshRendererAllocator, index);
    }

    template <> SpriteRenderer *getComponentByIndex<SpriteRenderer>(size_t index) const
    {
        return getComponentByIndex_impl(&mAllocators.mSpriteRendererAllocator, index);
    }

    template <> LineRenderer *getComponentByIndex<LineRenderer>(size_t index) const
    {
        return getComponentByIndex_impl(&mAllocators.mLineRendererAllocator, index);
    }

    template <> Rigidbody *getComponentByIndex<Rigidbody>(size_t index) const
    {
        return getComponentByIndex_impl(&mAllocators.mRigidbodyAllocator, index);
    }

    template <> Camera *getComponentByIndex<Camera>(size_t index) const
    {
        return getComponentByIndex_impl(&mAllocators.mCameraAllocator, index);
    }

    template <> Light *getComponentByIndex<Light>(size_t index) const
    {
        return getComponentByIndex_impl(&mAllocators.mLightAllocator, index);
    }

    template <> SphereCollider *getComponentByIndex<SphereCollider>(size_t index) const
    {
        return getComponentByIndex_impl(&mAllocators.mSphereColliderAllocator, index);
    }

    template <> BoxCollider *getComponentByIndex<BoxCollider>(size_t index) const
    {
        return getComponentByIndex_impl(&mAllocators.mBoxColliderAllocator, index);
    }

    template <> CapsuleCollider *getComponentByIndex<CapsuleCollider>(size_t index) const
    {
        return getComponentByIndex_impl(&mAllocators.mCapsuleColliderAllocator, index);
    }

    template <> MeshCollider *getComponentByIndex<MeshCollider>(size_t index) const
    {
        return getComponentByIndex_impl(&mAllocators.mMeshColliderAllocator, index);
    }

    template <> Transform *getComponentById<Transform>(const Guid &componentId) const
    {
        return getComponentById_impl(&mAllocators.mTransformAllocator, componentId);
    }

    template <> MeshRenderer *getComponentById<MeshRenderer>(const Guid &componentId) const
    {
        return getComponentById_impl(&mAllocators.mMeshRendererAllocator, componentId);
    }

    template <> SpriteRenderer *getComponentById<SpriteRenderer>(const Guid &componentId) const
    {
        return getComponentById_impl(&mAllocators.mSpriteRendererAllocator, componentId);
    }

    template <> LineRenderer *getComponentById<LineRenderer>(const Guid &componentId) const
    {
        return getComponentById_impl(&mAllocators.mLineRendererAllocator, componentId);
    }

    template <> Rigidbody *getComponentById<Rigidbody>(const Guid &componentId) const
    {
        return getComponentById_impl(&mAllocators.mRigidbodyAllocator, componentId);
    }

    template <> Camera *getComponentById<Camera>(const Guid &componentId) const
    {
        return getComponentById_impl(&mAllocators.mCameraAllocator, componentId);
    }

    template <> Light *getComponentById<Light>(const Guid &componentId) const
    {
        return getComponentById_impl(&mAllocators.mLightAllocator, componentId);
    }

    template <> SphereCollider *getComponentById<SphereCollider>(const Guid &componentId) const
    {
        return getComponentById_impl(&mAllocators.mSphereColliderAllocator, componentId);
    }

    template <> BoxCollider *getComponentById<BoxCollider>(const Guid &componentId) const
    {
        return getComponentById_impl(&mAllocators.mBoxColliderAllocator, componentId);
    }

    template <> CapsuleCollider *getComponentById<CapsuleCollider>(const Guid &componentId) const
    {
        return getComponentById_impl(&mAllocators.mCapsuleColliderAllocator, componentId);
    }

    template <> MeshCollider *getComponentById<MeshCollider>(const Guid &componentId) const
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

    template <> RenderTexture* createAsset<RenderTexture>()
    {
        return createAsset_impl(&mAllocators.mRenderTextureAllocator);
    }

    template <> Font *createAsset<Font>()
    {
        return createAsset_impl(&mAllocators.mFontAllocator);
    }

    template <> Sprite *createAsset<Sprite>()
    {
        return createAsset_impl(&mAllocators.mSpriteAllocator);
    }

  private:
    Asset *loadAssetFromYAML(const YAML::Node &in);
    Asset *loadAssetFromYAML(const YAML::Node &in, const Guid id, int type);
    Scene *loadSceneFromYAML(const YAML::Node &in);
    Scene *loadSceneFromYAML(const YAML::Node &in, const Guid id);

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

    template <typename T> T *getSystem_impl(const PoolAllocator<T> *allocator) const
    {
        static_assert(std::is_base_of<System, T>(), "'T' is not of type System");

        return allocator != nullptr ? allocator->get(0) : nullptr;
    }

    template <typename T> T *getComponent_impl(const PoolAllocator<T> *allocator, const Guid &entityId) const
    {
        static_assert(std::is_base_of<Component, T>(), "'T' is not of type Component");

        if (allocator == nullptr)
        {
            return nullptr;
        }

        std::vector<std::pair<Guid, int>> componentsOnEntity = getComponentsOnEntity(entityId);

        for (size_t i = 0; i < componentsOnEntity.size(); i++)
        {
            if (ComponentType<T>::type == componentsOnEntity[i].second)
            {
                std::unordered_map<Guid, int>::const_iterator it =
                    mIdState.mIdToGlobalIndex.find(componentsOnEntity[i].first);
                if (it != mIdState.mIdToGlobalIndex.end())
                {
                    return allocator->get(it->second);
                }

                break;
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

        T *component = allocator->construct(this, componentId);

        if (component != nullptr)
        {
            component->mEntityId = entityId;

            addIdToGlobalIndexMap_impl<T>(componentId, componentGlobalIndex, componentType);

            mIdState.mEntityIdToComponentIds[entityId].push_back(std::make_pair(componentId, componentType));

            mIdState.mComponentIdsMarkedCreated.push_back(std::make_tuple(entityId, componentId, componentType));
        }

        return component;
    }

    template <typename T> T *addSystem_impl(PoolAllocator<T> *allocator, size_t order)
    {
        static_assert(std::is_base_of<System, T>(), "'T' is not of type System");

        int systemGlobalIndex = (int)allocator->getCount();
        int systemType = SystemType<T>::type;
        Guid systemId = Guid::newGuid();

        T *system = allocator->construct(this, systemId);

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

    template <typename T> T *getSystemByIndex_impl(const PoolAllocator<T> *allocator, size_t index) const
    {
        static_assert(std::is_base_of<System, T>(), "'T' is not of type System");

        return allocator != nullptr ? allocator->get(index) : nullptr;
    }

    template <typename T> T *getSystemById_impl(const PoolAllocator<T> *allocator, const Guid &systemId) const
    {
        static_assert(std::is_base_of<System, T>(), "'T' is not of type System");

        if (allocator == nullptr || SystemType<T>::type != getTypeOf(systemId))
        {
            return nullptr;
        }

        return getById_impl<T>(mIdState.mIdToGlobalIndex, allocator, systemId);
    }

    template <typename T> T *getAssetByIndex_impl(const PoolAllocator<T> *allocator, size_t index) const
    {
        static_assert(std::is_base_of<Asset, T>(), "'T' is not of type Asset");

        return allocator != nullptr ? allocator->get(index) : nullptr;
    }

    template <typename T> T *getAssetById_impl(const PoolAllocator<T> *allocator, const Guid &assetId) const
    {
        static_assert(std::is_base_of<Asset, T>(), "'T' is not of type Asset");

        if (allocator == nullptr || AssetType<T>::type != getTypeOf(assetId))
        {
            return nullptr;
        }

        return getById_impl<T>(mIdState.mIdToGlobalIndex, allocator, assetId);
    }

    template <typename T> T *getComponentByIndex_impl(const PoolAllocator<T> *allocator, size_t index) const
    {
        static_assert(std::is_base_of<Component, T>(), "'T' is not of type Component");

        return allocator != nullptr ? allocator->get(index) : nullptr;
    }

    template <typename T> T *getComponentById_impl(const PoolAllocator<T> *allocator, const Guid &componentId) const
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

        T *asset = allocator->construct(this, id);

        if (asset != nullptr)
        {
            addIdToGlobalIndexMap_impl<T>(id, index, type);
        }

        return asset;
    }

    template <typename T> PoolAllocator<T> *getComponentAllocator_impl() const
    {
        static_assert(std::is_base_of<Component, T>(), "'T' is not of type Component");

        std::unordered_map<int, Allocator *>::const_iterator it =
            mAllocators.mComponentAllocatorMap.find(ComponentType<T>::type);
        if (it != mAllocators.mComponentAllocatorMap.end())
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
            mAllocators.mComponentAllocatorMap[ComponentType<T>::type] = allocator;
        }

        return allocator;
    }

    template <typename T> PoolAllocator<T> *getSystemAllocator_impl() const
    {
        static_assert(std::is_base_of<System, T>(), "'T' is not of type System");

        std::unordered_map<int, Allocator *>::const_iterator it =
            mAllocators.mSystemAllocatorMap.find(SystemType<T>::type);
        if (it != mAllocators.mSystemAllocatorMap.end())
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
            mAllocators.mSystemAllocatorMap[SystemType<T>::type] = allocator;
        }

        return allocator;
    }

    template <typename T> PoolAllocator<T> *getAssetAllocator_impl() const
    {
        static_assert(std::is_base_of<Asset, T>(), "'T' is not of type Asset");

        std::unordered_map<int, Allocator *>::const_iterator it =
            mAllocators.mAssetAllocatorMap.find(AssetType<T>::type);
        if (it != mAllocators.mAssetAllocatorMap.end())
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
            mAllocators.mAssetAllocatorMap[AssetType<T>::type] = allocator;
        }

        return allocator;
    }

    template <typename T>
    T *getById_impl(const std::unordered_map<Guid, int> &idToIndexMap, const PoolAllocator<T> *allocator,
                    const Guid &id) const
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

    // Explicit specializations

    template <>
    Transform *getComponentById_impl<Transform>(const PoolAllocator<Transform> *allocator,
                                                const Guid &componentId) const
    {
        return getById_impl<Transform>(mIdState.mTransformIdToGlobalIndex, allocator, componentId);
    }

    template <>
    MeshRenderer *getComponentById_impl<MeshRenderer>(const PoolAllocator<MeshRenderer> *allocator,
                                                      const Guid &componentId) const
    {
        return getById_impl<MeshRenderer>(mIdState.mMeshRendererIdToGlobalIndex, allocator, componentId);
    }

    template <>
    SpriteRenderer *getComponentById_impl<SpriteRenderer>(const PoolAllocator<SpriteRenderer> *allocator,
                                                          const Guid &componentId) const
    {
        return getById_impl<SpriteRenderer>(mIdState.mSpriteRendererIdToGlobalIndex, allocator, componentId);
    }

    template <>
    LineRenderer *getComponentById_impl<LineRenderer>(const PoolAllocator<LineRenderer> *allocator,
                                                      const Guid &componentId) const
    {
        return getById_impl<LineRenderer>(mIdState.mLineRendererIdToGlobalIndex, allocator, componentId);
    }

    template <>
    Rigidbody *getComponentById_impl<Rigidbody>(const PoolAllocator<Rigidbody> *allocator,
                                                const Guid &componentId) const
    {
        return getById_impl<Rigidbody>(mIdState.mRigidbodyIdToGlobalIndex, allocator, componentId);
    }

    template <>
    Camera *getComponentById_impl<Camera>(const PoolAllocator<Camera> *allocator, const Guid &componentId) const
    {
        return getById_impl<Camera>(mIdState.mCameraIdToGlobalIndex, allocator, componentId);
    }

    template <>
    Light *getComponentById_impl<Light>(const PoolAllocator<Light> *allocator, const Guid &componentId) const
    {
        return getById_impl<Light>(mIdState.mLightIdToGlobalIndex, allocator, componentId);
    }

    template <>
    SphereCollider *getComponentById_impl<SphereCollider>(const PoolAllocator<SphereCollider> *allocator,
                                                          const Guid &componentId) const
    {
        return getById_impl<SphereCollider>(mIdState.mSphereColliderIdToGlobalIndex, allocator, componentId);
    }

    template <>
    BoxCollider *getComponentById_impl<BoxCollider>(const PoolAllocator<BoxCollider> *allocator,
                                                    const Guid &componentId) const
    {
        return getById_impl<BoxCollider>(mIdState.mBoxColliderIdToGlobalIndex, allocator, componentId);
    }

    template <>
    CapsuleCollider *getComponentById_impl<CapsuleCollider>(const PoolAllocator<CapsuleCollider> *allocator,
                                                            const Guid &componentId) const
    {
        return getById_impl<CapsuleCollider>(mIdState.mCapsuleColliderIdToGlobalIndex, allocator, componentId);
    }

    template <>
    MeshCollider *getComponentById_impl<MeshCollider>(const PoolAllocator<MeshCollider> *allocator,
                                                      const Guid &componentId) const
    {
        return getById_impl<MeshCollider>(mIdState.mMeshColliderIdToGlobalIndex, allocator, componentId);
    }

    template <> Mesh *getAssetById_impl<Mesh>(const PoolAllocator<Mesh> *allocator, const Guid &assetId) const
    {
        return getById_impl<Mesh>(mIdState.mMeshIdToGlobalIndex, allocator, assetId);
    }

    template <>
    Material *getAssetById_impl<Material>(const PoolAllocator<Material> *allocator, const Guid &assetId) const
    {
        return getById_impl<Material>(mIdState.mMaterialIdToGlobalIndex, allocator, assetId);
    }

    template <> Shader *getAssetById_impl<Shader>(const PoolAllocator<Shader> *allocator, const Guid &assetId) const
    {
        return getById_impl<Shader>(mIdState.mShaderIdToGlobalIndex, allocator, assetId);
    }

    template <>
    Texture2D *getAssetById_impl<Texture2D>(const PoolAllocator<Texture2D> *allocator, const Guid &assetId) const
    {
        return getById_impl<Texture2D>(mIdState.mTexture2DIdToGlobalIndex, allocator, assetId);
    }

    template <>
    Texture3D *getAssetById_impl<Texture3D>(const PoolAllocator<Texture3D> *allocator, const Guid &assetId) const
    {
        return getById_impl<Texture3D>(mIdState.mTexture3DIdToGlobalIndex, allocator, assetId);
    }

    template <> Cubemap *getAssetById_impl<Cubemap>(const PoolAllocator<Cubemap> *allocator, const Guid &assetId) const
    {
        return getById_impl<Cubemap>(mIdState.mCubemapIdToGlobalIndex, allocator, assetId);
    }

    template <> RenderTexture* getAssetById_impl<RenderTexture>(const PoolAllocator<RenderTexture>* allocator, const Guid& assetId) const
    {
        return getById_impl<RenderTexture>(mIdState.mRenderTextureIdToGlobalIndex, allocator, assetId);
    }

    template <> Font *getAssetById_impl<Font>(const PoolAllocator<Font> *allocator, const Guid &assetId) const
    {
        return getById_impl<Font>(mIdState.mFontIdToGlobalIndex, allocator, assetId);
    }

    template <> Sprite *getAssetById_impl<Sprite>(const PoolAllocator<Sprite> *allocator, const Guid &assetId) const
    {
        return getById_impl<Sprite>(mIdState.mSpriteIdToGlobalIndex, allocator, assetId);
    }

    template <>
    RenderSystem *getSystemById_impl<RenderSystem>(const PoolAllocator<RenderSystem> *allocator,
                                                   const Guid &assetId) const
    {
        return getById_impl<RenderSystem>(mIdState.mRenderSystemIdToGlobalIndex, allocator, assetId);
    }

    template <>
    PhysicsSystem *getSystemById_impl<PhysicsSystem>(const PoolAllocator<PhysicsSystem> *allocator,
                                                     const Guid &assetId) const
    {
        return getById_impl<PhysicsSystem>(mIdState.mPhysicsSystemIdToGlobalIndex, allocator, assetId);
    }

    template <>
    CleanUpSystem *getSystemById_impl<CleanUpSystem>(const PoolAllocator<CleanUpSystem> *allocator,
                                                     const Guid &assetId) const
    {
        return getById_impl<CleanUpSystem>(mIdState.mCleanupSystemIdToGlobalIndex, allocator, assetId);
    }

    template <>
    DebugSystem *getSystemById_impl<DebugSystem>(const PoolAllocator<DebugSystem> *allocator, const Guid &assetId) const
    {
        return getById_impl<DebugSystem>(mIdState.mDebugSystemIdToGlobalIndex, allocator, assetId);
    }

    template <>
    GizmoSystem *getSystemById_impl<GizmoSystem>(const PoolAllocator<GizmoSystem> *allocator, const Guid &assetId) const
    {
        return getById_impl<GizmoSystem>(mIdState.mGizmoSystemIdToGlobalIndex, allocator, assetId);
    }

    template <>
    FreeLookCameraSystem* getSystemById_impl<FreeLookCameraSystem>(const PoolAllocator<FreeLookCameraSystem>* allocator, const Guid& assetId) const
    {
        return getById_impl<FreeLookCameraSystem>(mIdState.mFreeLookCameraSystemIdToGlobalIndex, allocator, assetId);
    }

    template <> void addIdToGlobalIndexMap_impl<Scene>(const Guid &id, int index, int type)
    {
        mIdState.mSceneIdToGlobalIndex[id] = index;
        mIdState.mIdToGlobalIndex[id] = index;
        mIdState.mIdToType[id] = type;
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

    template <> void addIdToGlobalIndexMap_impl<SpriteRenderer>(const Guid &id, int index, int type)
    {
        mIdState.mSpriteRendererIdToGlobalIndex[id] = index;
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

    template <> void addIdToGlobalIndexMap_impl<RenderTexture>(const Guid& id, int index, int type)
    {
        mIdState.mRenderTextureIdToGlobalIndex[id] = index;
        mIdState.mIdToGlobalIndex[id] = index;
        mIdState.mIdToType[id] = type;
    }

    template <> void addIdToGlobalIndexMap_impl<Font>(const Guid &id, int index, int type)
    {
        mIdState.mFontIdToGlobalIndex[id] = index;
        mIdState.mIdToGlobalIndex[id] = index;
        mIdState.mIdToType[id] = type;
    }

    template <> void addIdToGlobalIndexMap_impl<Sprite>(const Guid &id, int index, int type)
    {
        mIdState.mSpriteIdToGlobalIndex[id] = index;
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

    template <> void addIdToGlobalIndexMap_impl<FreeLookCameraSystem>(const Guid& id, int index, int type)
    {
        mIdState.mFreeLookCameraSystemIdToGlobalIndex[id] = index;
        mIdState.mIdToGlobalIndex[id] = index;
        mIdState.mIdToType[id] = type;
    }
};
} // namespace PhysicsEngine

#endif