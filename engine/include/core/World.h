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
#include "SerializationInternal.h"
#include "Shader.h"
#include "Texture2D.h"
#include "Texture3D.h"
#include "Util.h"
#include "WorldUtil.h"

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
#include "../systems/GizmoSystem.h"
#include "../systems/System.h"

namespace PhysicsEngine
{
class World
{
  private:
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

    // internal world entity id state
    std::unordered_map<Guid, int> mEntityIdToGlobalIndex;

    // internal world components id state
    std::unordered_map<Guid, int> mTransformIdToGlobalIndex;
    std::unordered_map<Guid, int> mMeshRendererIdToGlobalIndex;
    std::unordered_map<Guid, int> mLineRendererIdToGlobalIndex;
    std::unordered_map<Guid, int> mRigidbodyIdToGlobalIndex;
    std::unordered_map<Guid, int> mCameraIdToGlobalIndex;
    std::unordered_map<Guid, int> mLightIdToGlobalIndex;
    std::unordered_map<Guid, int> mSphereColliderIdToGlobalIndex;
    std::unordered_map<Guid, int> mBoxColliderIdToGlobalIndex;
    std::unordered_map<Guid, int> mCapsuleColliderIdToGlobalIndex;
    std::unordered_map<Guid, int> mMeshColliderIdToGlobalIndex;

    // internal world asset id state
    std::unordered_map<Guid, int> mMeshIdToGlobalIndex;
    std::unordered_map<Guid, int> mMaterialIdToGlobalIndex;
    std::unordered_map<Guid, int> mShaderIdToGlobalIndex;
    std::unordered_map<Guid, int> mTexture2DIdToGlobalIndex;
    std::unordered_map<Guid, int> mTexture3DIdToGlobalIndex;
    std::unordered_map<Guid, int> mCubemapIdToGlobalIndex;
    std::unordered_map<Guid, int> mFontIdToGlobalIndex;

    // internal world system id state
    std::unordered_map<Guid, int> mRenderSystemIdToGlobalIndex;
    std::unordered_map<Guid, int> mPhysicsSystemIdToGlobalIndex;
    std::unordered_map<Guid, int> mCleanupSystemIdToGlobalIndex;
    std::unordered_map<Guid, int> mDebugSystemIdToGlobalIndex;
    std::unordered_map<Guid, int> mGizmoSystemIdToGlobalIndex;

    // world id state for all entity, components, systems, and assets
    std::unordered_map<Guid, int> mIdToGlobalIndex;
    std::unordered_map<Guid, int> mIdToType;

    // entity ids to component ids
    std::unordered_map<Guid, std::vector<std::pair<Guid, int>>> mEntityIdToComponentIds;

    // entity creation/deletion state
    std::vector<Guid> mEntityIdsMarkedCreated;
    std::vector<Guid> mEntityIdsMarkedLatentDestroy;

    // component create/deletion state
    std::vector<triple<Guid, Guid, int>> mComponentIdsMarkedCreated;
    std::vector<triple<Guid, Guid, int>> mComponentIdsMarkedLatentDestroy;

    // all systems in world listed in order they should be updated
    std::vector<System *> mSystems;

    // asset and scene id to filepath
    std::unordered_map<Guid, std::string> mAssetIdToFilepath;
    std::unordered_map<Guid, std::string> mSceneIdToFilepath;

    // default loaded meshes
    Guid mSphereMeshId;
    Guid mCubeMeshId;

    // default loaded shaders
    Guid mFontShaderId;
    Guid mGizmoShaderId;
    Guid mColorShaderId;
    Guid mPositionAndNormalsShaderId;
    Guid mSsaoShaderId;
    Guid mScreenQuadShaderId;
    Guid mNormalMapShaderId;
    Guid mDepthMapShaderId;
    Guid mShadowDepthMapShaderId;
    Guid mShadowDepthCubemapShaderId;
    Guid mGbufferShaderId;
    Guid mSimpleLitShaderId;
    Guid mSimpleLitDeferedShaderId;
    Guid mOverdrawShaderId;

    // default loaded materials
    Guid mSimpleLitMaterialId;
    Guid mColorMaterialId;

  public:
    World();
    ~World();
    World(const World &other) = delete;
    World &operator=(const World &other) = delete;

    bool loadAsset(const std::string &filePath);
    bool loadScene(const std::string &filePath, bool ignoreSystems = false);
    bool loadSceneFromEditor(const std::string &filePath);

    void latentDestroyEntitiesInWorld();

    int getNumberOfEntities() const;
    int getNumberOfUpdatingSystems() const;

    template <typename T> int getNumberOfSystems() const
    {
        static_assert(IsSystem<T>::value == true, "'T' is not of type System");

        return getNumberOfSystems_impl<T>(getSystemAllocator_impl<T>());
    }

    template <typename T> int getNumberOfComponents() const
    {
        static_assert(IsComponent<T>::value == true, "'T' is not of type Component");

        return getNumberOfComponents_impl<T>(getComponentAllocator_impl<T>());
    }

    template <typename T> int getNumberOfAssets() const
    {
        static_assert(IsAsset<T>::value == true, "'T' is not of type Asset");

        return getNumberOfAssets_impl(getAssetAllocator_impl<T>());
    }

    template <typename T> T *getSystem()
    {
        static_assert(IsSystem<T>::value == true, "'T' is not of type System");

        return getSystem_impl(getSystemAllocator_impl<T>());
    }

    template <typename T> T *getComponent(const Guid &entityId)
    {
        static_assert(IsComponent<T>::value == true, "'T' is not of type Component");

        return getComponent_impl<T>(getComponentAllocator_impl<T>(), entityId);
    }

    template <typename T> T *addComponent(const Guid &entityId)
    {
        static_assert(IsComponent<T>::value == true, "'T' is not of type Component");

        return addComponent_impl<T>(getComponentOrAddAllocator_impl<T>(), entityId);
    }

    template <typename T> T *addComponent(const std::vector<char> &data)
    {
        static_assert(IsComponent<T>::value == true, "'T' is not of type Component");

        return addComponent_impl<T>(getComponentOrAddAllocator_impl<T>(), data);
    }

    template <typename T> T *addSystem(int order)
    {
        static_assert(IsSystem<T>::value == true, "'T' is not of type System");

        return addSystem_impl<T>(getSystemOrAddAllocator_impl<T>(), order);
    }

    Entity *getEntityByIndex(int index);
    Entity *getEntityById(const Guid &entityId);
    System *getSystemByUpdateOrder(int order);

    template <typename T> T *getSystemByIndex(int index)
    {
        static_assert(IsSystem<T>::value == true, "'T' is not of type System");

        return getSystemByIndex_impl(getSystemAllocator_impl<T>(), index);
    }

    template <typename T> T *getSystemById(const Guid &systemId)
    {
        static_assert(IsSystem<T>::value == true, "'T' is not of type System");

        return getSystemById_impl<T>(getSystemAllocator_impl<T>(), systemId);
    }

    template <typename T> T *getAssetByIndex(int index)
    {
        static_assert(IsAsset<T>::value == true, "'T' is not of type Asset");

        return getAssetByIndex_impl(getAssetAllocator_impl<T>(), index);
    }

    template <typename T> T *getAssetById(const Guid &assetId)
    {
        static_assert(IsAsset<T>::value == true, "'T' is not of type Asset");

        return getAssetById_impl<T>(getAssetAllocator_impl<T>(), assetId);
    }

    template <typename T> T *getComponentByIndex(int index)
    {
        static_assert(IsComponent<T>::value == true, "'T' is not of type Component");

        return getComponentByIndex_impl(getComponentAllocator_impl<T>(), index);
    }

    template <typename T> T *getComponentById(const Guid &componentId)
    {
        static_assert(IsComponent<T>::value == true, "'T' is not of type Component");

        return getComponentById_impl<T>(getComponentAllocator_impl<T>(), componentId);
    }

    template <typename T> T *createAsset()
    {
        static_assert(IsAsset<T>::value == true, "'T' is not of type Asset");

        return createAsset_impl<T>(getAssetOrAddAllocator_impl<T>());
    }

    template <typename T> T *createAsset(const std::vector<char> &data)
    {
        static_assert(IsAsset<T>::value == true, "'T' is not of type Asset");

        return createAsset_impl<T>(getAssetOrAddAllocator_impl<T>(), data);
    }

    int getIndexOf(const Guid &id) const;
    int getTypeOf(const Guid &id) const;

    Entity *createEntity();
    Entity *createEntity(const std::vector<char> &data);

    void latentDestroyEntity(const Guid &entityId);
    void immediateDestroyEntity(const Guid &entityId);
    void latentDestroyComponent(const Guid &entityId, const Guid &componentId, int componentType);
    void immediateDestroyComponent(const Guid &entityId, const Guid &componentId, int componentType);
    bool isMarkedForLatentDestroy(const Guid &id);
    void clearIdsMarkedCreatedOrDestroyed();

    std::vector<std::pair<Guid, int>> getComponentsOnEntity(const Guid &entityId);

    std::vector<Guid> getEntityIdsMarkedCreated() const;
    std::vector<Guid> getEntityIdsMarkedLatentDestroy() const;
    std::vector<triple<Guid, Guid, int>> getComponentIdsMarkedCreated() const;
    std::vector<triple<Guid, Guid, int>> getComponentIdsMarkedLatentDestroy() const;

    std::string getAssetFilepath(const Guid &assetId) const;
    std::string getSceneFilepath(const Guid &sceneId) const;

    // Explicit template specializations

    template <> int getNumberOfSystems<RenderSystem>() const
    {
        return (int)mRenderSystemAllocator.getCount();
    }

    template <> int getNumberOfSystems<PhysicsSystem>() const
    {
        return (int)mPhysicsSystemAllocator.getCount();
    }

    template <> int getNumberOfSystems<CleanUpSystem>() const
    {
        return (int)mCleanupSystemAllocator.getCount();
    }

    template <> int getNumberOfSystems<DebugSystem>() const
    {
        return (int)mDebugSystemAllocator.getCount();
    }

    template <> int getNumberOfSystems<GizmoSystem>() const
    {
        return (int)mGizmoSystemAllocator.getCount();
    }

    template <> int getNumberOfComponents<Transform>() const
    {
        return (int)mTransformAllocator.getCount();
    }

    template <> int getNumberOfComponents<MeshRenderer>() const
    {
        return (int)mMeshRendererAllocator.getCount();
    }

    template <> int getNumberOfComponents<LineRenderer>() const
    {
        return (int)mLineRendererAllocator.getCount();
    }

    template <> int getNumberOfComponents<Rigidbody>() const
    {
        return (int)mRigidbodyAllocator.getCount();
    }

    template <> int getNumberOfComponents<Camera>() const
    {
        return (int)mCameraAllocator.getCount();
    }

    template <> int getNumberOfComponents<Light>() const
    {
        return (int)mLightAllocator.getCount();
    }

    template <> int getNumberOfComponents<SphereCollider>() const
    {
        return (int)mSphereColliderAllocator.getCount();
    }

    template <> int getNumberOfComponents<BoxCollider>() const
    {
        return (int)mBoxColliderAllocator.getCount();
    }

    template <> int getNumberOfComponents<CapsuleCollider>() const
    {
        return (int)mCapsuleColliderAllocator.getCount();
    }

    template <> int getNumberOfComponents<MeshCollider>() const
    {
        return (int)mMeshColliderAllocator.getCount();
    }

    template <> int getNumberOfAssets<Mesh>() const
    {
        return (int)mMeshAllocator.getCount();
    }

    template <> int getNumberOfAssets<Material>() const
    {
        return (int)mMaterialAllocator.getCount();
    }

    template <> int getNumberOfAssets<Shader>() const
    {
        return (int)mShaderAllocator.getCount();
    }

    template <> int getNumberOfAssets<Texture2D>() const
    {
        return (int)mTexture2DAllocator.getCount();
    }

    template <> int getNumberOfAssets<Texture3D>() const
    {
        return (int)mTexture3DAllocator.getCount();
    }

    template <> int getNumberOfAssets<Cubemap>() const
    {
        return (int)mCubemapAllocator.getCount();
    }

    template <> int getNumberOfAssets<Font>() const
    {
        return (int)mFontAllocator.getCount();
    }

    template <> RenderSystem *getSystem<RenderSystem>()
    {
        return getSystem_impl(&mRenderSystemAllocator);
    }

    template <> PhysicsSystem *getSystem<PhysicsSystem>()
    {
        return getSystem_impl(&mPhysicsSystemAllocator);
    }

    template <> CleanUpSystem *getSystem<CleanUpSystem>()
    {
        return getSystem_impl(&mCleanupSystemAllocator);
    }

    template <> DebugSystem *getSystem<DebugSystem>()
    {
        return getSystem_impl(&mDebugSystemAllocator);
    }

    template <> GizmoSystem* getSystem<GizmoSystem>()
    {
        return getSystem_impl(&mGizmoSystemAllocator);
    }

    template <> Transform *getComponent<Transform>(const Guid &entityId)
    {
        return getComponent_impl(&mTransformAllocator, entityId);
    }

    template <> MeshRenderer *getComponent<MeshRenderer>(const Guid &entityId)
    {
        return getComponent_impl(&mMeshRendererAllocator, entityId);
    }

    template <> LineRenderer *getComponent<LineRenderer>(const Guid &entityId)
    {
        return getComponent_impl(&mLineRendererAllocator, entityId);
    }

    template <> Rigidbody *getComponent<Rigidbody>(const Guid &entityId)
    {
        return getComponent_impl(&mRigidbodyAllocator, entityId);
    }

    template <> Camera *getComponent<Camera>(const Guid &entityId)
    {
        return getComponent_impl(&mCameraAllocator, entityId);
    }

    template <> Light *getComponent<Light>(const Guid &entityId)
    {
        return getComponent_impl(&mLightAllocator, entityId);
    }

    template <> SphereCollider *getComponent<SphereCollider>(const Guid &entityId)
    {
        return getComponent_impl(&mSphereColliderAllocator, entityId);
    }

    template <> BoxCollider *getComponent<BoxCollider>(const Guid &entityId)
    {
        return getComponent_impl(&mBoxColliderAllocator, entityId);
    }

    template <> CapsuleCollider *getComponent<CapsuleCollider>(const Guid &entityId)
    {
        return getComponent_impl(&mCapsuleColliderAllocator, entityId);
    }

    template <> MeshCollider *getComponent<MeshCollider>(const Guid &entityId)
    {
        return getComponent_impl(&mMeshColliderAllocator, entityId);
    }

    template <> Transform *addComponent<Transform>(const Guid &entityId)
    {
        return addComponent_impl(&mTransformAllocator, entityId);
    }

    template <> MeshRenderer *addComponent<MeshRenderer>(const Guid &entityId)
    {
        return addComponent_impl(&mMeshRendererAllocator, entityId);
    }

    template <> LineRenderer *addComponent<LineRenderer>(const Guid &entityId)
    {
        return addComponent_impl(&mLineRendererAllocator, entityId);
    }

    template <> Rigidbody *addComponent<Rigidbody>(const Guid &entityId)
    {
        return addComponent_impl(&mRigidbodyAllocator, entityId);
    }

    template <> Camera *addComponent<Camera>(const Guid &entityId)
    {
        return addComponent_impl(&mCameraAllocator, entityId);
    }

    template <> Light *addComponent<Light>(const Guid &entityId)
    {
        return addComponent_impl(&mLightAllocator, entityId);
    }

    template <> SphereCollider *addComponent<SphereCollider>(const Guid &entityId)
    {
        return addComponent_impl(&mSphereColliderAllocator, entityId);
    }

    template <> BoxCollider *addComponent<BoxCollider>(const Guid &entityId)
    {
        return addComponent_impl(&mBoxColliderAllocator, entityId);
    }

    template <> CapsuleCollider *addComponent<CapsuleCollider>(const Guid &entityId)
    {
        return addComponent_impl(&mCapsuleColliderAllocator, entityId);
    }

    template <> MeshCollider *addComponent<MeshCollider>(const Guid &entityId)
    {
        return addComponent_impl(&mMeshColliderAllocator, entityId);
    }

    template <> Transform *addComponent<Transform>(const std::vector<char> &data)
    {
        return addComponent_impl(&mTransformAllocator, data);
    }

    template <> MeshRenderer *addComponent<MeshRenderer>(const std::vector<char> &data)
    {
        return addComponent_impl(&mMeshRendererAllocator, data);
    }

    template <> LineRenderer *addComponent<LineRenderer>(const std::vector<char> &data)
    {
        return addComponent_impl(&mLineRendererAllocator, data);
    }

    template <> Rigidbody *addComponent<Rigidbody>(const std::vector<char> &data)
    {
        return addComponent_impl(&mRigidbodyAllocator, data);
    }

    template <> Camera *addComponent<Camera>(const std::vector<char> &data)
    {
        return addComponent_impl(&mCameraAllocator, data);
    }

    template <> Light *addComponent<Light>(const std::vector<char> &data)
    {
        return addComponent_impl(&mLightAllocator, data);
    }

    template <> SphereCollider *addComponent<SphereCollider>(const std::vector<char> &data)
    {
        return addComponent_impl(&mSphereColliderAllocator, data);
    }

    template <> BoxCollider *addComponent<BoxCollider>(const std::vector<char> &data)
    {
        return addComponent_impl(&mBoxColliderAllocator, data);
    }

    template <> CapsuleCollider *addComponent<CapsuleCollider>(const std::vector<char> &data)
    {
        return addComponent_impl(&mCapsuleColliderAllocator, data);
    }

    template <> MeshCollider *addComponent<MeshCollider>(const std::vector<char> &data)
    {
        return addComponent_impl(&mMeshColliderAllocator, data);
    }

    template <> RenderSystem *addSystem<RenderSystem>(int order)
    {
        return addSystem_impl(&mRenderSystemAllocator, order);
    }

    template <> PhysicsSystem *addSystem<PhysicsSystem>(int order)
    {
        return addSystem_impl(&mPhysicsSystemAllocator, order);
    }

    template <> CleanUpSystem *addSystem<CleanUpSystem>(int order)
    {
        return addSystem_impl(&mCleanupSystemAllocator, order);
    }

    template <> DebugSystem *addSystem<DebugSystem>(int order)
    {
        return addSystem_impl(&mDebugSystemAllocator, order);
    }

    template <> GizmoSystem* addSystem<GizmoSystem>(int order)
    {
        return addSystem_impl(&mGizmoSystemAllocator, order);
    }

    template <> RenderSystem *getSystemByIndex<RenderSystem>(int index)
    {
        return getSystemByIndex_impl(&mRenderSystemAllocator, index);
    }

    template <> PhysicsSystem *getSystemByIndex<PhysicsSystem>(int index)
    {
        return getSystemByIndex_impl(&mPhysicsSystemAllocator, index);
    }

    template <> CleanUpSystem *getSystemByIndex<CleanUpSystem>(int index)
    {
        return getSystemByIndex_impl(&mCleanupSystemAllocator, index);
    }

    template <> DebugSystem *getSystemByIndex<DebugSystem>(int index)
    {
        return getSystemByIndex_impl(&mDebugSystemAllocator, index);
    }

    template <> GizmoSystem* getSystemByIndex<GizmoSystem>(int index)
    {
        return getSystemByIndex_impl(&mGizmoSystemAllocator, index);
    }

    template <> RenderSystem *getSystemById<RenderSystem>(const Guid &systemId)
    {
        return getSystemById_impl(&mRenderSystemAllocator, systemId);
    }

    template <> PhysicsSystem *getSystemById<PhysicsSystem>(const Guid &systemId)
    {
        return getSystemById_impl(&mPhysicsSystemAllocator, systemId);
    }

    template <> CleanUpSystem *getSystemById<CleanUpSystem>(const Guid &systemId)
    {
        return getSystemById_impl(&mCleanupSystemAllocator, systemId);
    }

    template <> DebugSystem *getSystemById<DebugSystem>(const Guid &systemId)
    {
        return getSystemById_impl(&mDebugSystemAllocator, systemId);
    }

    template <> GizmoSystem* getSystemById<GizmoSystem>(const Guid& systemId)
    {
        return getSystemById_impl(&mGizmoSystemAllocator, systemId);
    }

    template <> Mesh *getAssetByIndex<Mesh>(int index)
    {
        return getAssetByIndex_impl(&mMeshAllocator, index);
    }

    template <> Material *getAssetByIndex<Material>(int index)
    {
        return getAssetByIndex_impl(&mMaterialAllocator, index);
    }

    template <> Shader *getAssetByIndex<Shader>(int index)
    {
        return getAssetByIndex_impl(&mShaderAllocator, index);
    }

    template <> Texture2D *getAssetByIndex<Texture2D>(int index)
    {
        return getAssetByIndex_impl(&mTexture2DAllocator, index);
    }

    template <> Texture3D *getAssetByIndex<Texture3D>(int index)
    {
        return getAssetByIndex_impl(&mTexture3DAllocator, index);
    }

    template <> Cubemap *getAssetByIndex<Cubemap>(int index)
    {
        return getAssetByIndex_impl(&mCubemapAllocator, index);
    }

    template <> Font *getAssetByIndex<Font>(int index)
    {
        return getAssetByIndex_impl(&mFontAllocator, index);
    }

    template <> Mesh *getAssetById<Mesh>(const Guid &assetId)
    {
        return getAssetById_impl(&mMeshAllocator, assetId);
    }

    template <> Material *getAssetById<Material>(const Guid &assetId)
    {
        return getAssetById_impl(&mMaterialAllocator, assetId);
    }

    template <> Shader *getAssetById<Shader>(const Guid &assetId)
    {
        return getAssetById_impl(&mShaderAllocator, assetId);
    }

    template <> Texture2D *getAssetById<Texture2D>(const Guid &assetId)
    {
        return getAssetById_impl(&mTexture2DAllocator, assetId);
    }

    template <> Texture3D *getAssetById<Texture3D>(const Guid &assetId)
    {
        return getAssetById_impl(&mTexture3DAllocator, assetId);
    }

    template <> Cubemap *getAssetById<Cubemap>(const Guid &assetId)
    {
        return getAssetById_impl(&mCubemapAllocator, assetId);
    }

    template <> Font *getAssetById<Font>(const Guid &assetId)
    {
        return getAssetById_impl(&mFontAllocator, assetId);
    }

    template <> Transform *getComponentByIndex<Transform>(int index)
    {
        return getComponentByIndex_impl(&mTransformAllocator, index);
    }

    template <> MeshRenderer *getComponentByIndex<MeshRenderer>(int index)
    {
        return getComponentByIndex_impl(&mMeshRendererAllocator, index);
    }

    template <> LineRenderer *getComponentByIndex<LineRenderer>(int index)
    {
        return getComponentByIndex_impl(&mLineRendererAllocator, index);
    }

    template <> Rigidbody *getComponentByIndex<Rigidbody>(int index)
    {
        return getComponentByIndex_impl(&mRigidbodyAllocator, index);
    }

    template <> Camera *getComponentByIndex<Camera>(int index)
    {
        return getComponentByIndex_impl(&mCameraAllocator, index);
    }

    template <> Light *getComponentByIndex<Light>(int index)
    {
        return getComponentByIndex_impl(&mLightAllocator, index);
    }

    template <> SphereCollider *getComponentByIndex<SphereCollider>(int index)
    {
        return getComponentByIndex_impl(&mSphereColliderAllocator, index);
    }

    template <> BoxCollider *getComponentByIndex<BoxCollider>(int index)
    {
        return getComponentByIndex_impl(&mBoxColliderAllocator, index);
    }

    template <> CapsuleCollider *getComponentByIndex<CapsuleCollider>(int index)
    {
        return getComponentByIndex_impl(&mCapsuleColliderAllocator, index);
    }

    template <> MeshCollider *getComponentByIndex<MeshCollider>(int index)
    {
        return getComponentByIndex_impl(&mMeshColliderAllocator, index);
    }

    template <> Transform *getComponentById<Transform>(const Guid &componentId)
    {
        return getComponentById_impl(&mTransformAllocator, componentId);
    }

    template <> MeshRenderer *getComponentById<MeshRenderer>(const Guid &componentId)
    {
        return getComponentById_impl(&mMeshRendererAllocator, componentId);
    }

    template <> LineRenderer *getComponentById<LineRenderer>(const Guid &componentId)
    {
        return getComponentById_impl(&mLineRendererAllocator, componentId);
    }

    template <> Rigidbody *getComponentById<Rigidbody>(const Guid &componentId)
    {
        return getComponentById_impl(&mRigidbodyAllocator, componentId);
    }

    template <> Camera *getComponentById<Camera>(const Guid &componentId)
    {
        return getComponentById_impl(&mCameraAllocator, componentId);
    }

    template <> Light *getComponentById<Light>(const Guid &componentId)
    {
        return getComponentById_impl(&mLightAllocator, componentId);
    }

    template <> SphereCollider *getComponentById<SphereCollider>(const Guid &componentId)
    {
        return getComponentById_impl(&mSphereColliderAllocator, componentId);
    }

    template <> BoxCollider *getComponentById<BoxCollider>(const Guid &componentId)
    {
        return getComponentById_impl(&mBoxColliderAllocator, componentId);
    }

    template <> CapsuleCollider *getComponentById<CapsuleCollider>(const Guid &componentId)
    {
        return getComponentById_impl(&mCapsuleColliderAllocator, componentId);
    }

    template <> MeshCollider *getComponentById<MeshCollider>(const Guid &componentId)
    {
        return getComponentById_impl(&mMeshColliderAllocator, componentId);
    }

    template <> Mesh *createAsset<Mesh>()
    {
        return createAsset_impl(&mMeshAllocator);
    }

    template <> Material *createAsset<Material>()
    {
        return createAsset_impl(&mMaterialAllocator);
    }

    template <> Shader *createAsset<Shader>()
    {
        return createAsset_impl(&mShaderAllocator);
    }

    template <> Texture2D *createAsset<Texture2D>()
    {
        return createAsset_impl(&mTexture2DAllocator);
    }

    template <> Texture3D *createAsset<Texture3D>()
    {
        return createAsset_impl(&mTexture3DAllocator);
    }

    template <> Cubemap *createAsset<Cubemap>()
    {
        return createAsset_impl(&mCubemapAllocator);
    }

    template <> Font *createAsset<Font>()
    {
        return createAsset_impl(&mFontAllocator);
    }

    template <> Mesh *createAsset(const std::vector<char> &data)
    {
        return createAsset_impl(&mMeshAllocator, data);
    }

    template <> Material *createAsset(const std::vector<char> &data)
    {
        return createAsset_impl(&mMaterialAllocator, data);
    }

    template <> Shader *createAsset(const std::vector<char> &data)
    {
        return createAsset_impl(&mShaderAllocator, data);
    }

    template <> Texture2D *createAsset(const std::vector<char> &data)
    {
        return createAsset_impl(&mTexture2DAllocator, data);
    }

    template <> Texture3D *createAsset(const std::vector<char> &data)
    {
        return createAsset_impl(&mTexture3DAllocator, data);
    }

    template <> Cubemap *createAsset(const std::vector<char> &data)
    {
        return createAsset_impl(&mCubemapAllocator, data);
    }

    template <> Font *createAsset(const std::vector<char> &data)
    {
        return createAsset_impl(&mFontAllocator, data);
    }

    // default asset getters
    Guid getSphereMesh() const;
    Guid getCubeMesh() const;
    Guid getColorMaterial() const;
    Guid getSimpleLitMaterial() const;
    Guid getFontShaderId() const;
    Guid getGizmoShaderId() const;
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
    template <typename T> int getNumberOfSystems_impl(const PoolAllocator<T> *allocator) const
    {
        static_assert(IsSystem<T>::value == true, "'T' is not of type System");

        return allocator != NULL ? (int)allocator->getCount() : 0;
    }

    template <typename T> int getNumberOfComponents_impl(const PoolAllocator<T> *allocator) const
    {
        static_assert(IsComponent<T>::value == true, "'T' is not of type Component");

        return allocator != NULL ? (int)allocator->getCount() : 0;
    }

    template <typename T> int getNumberOfAssets_impl(const PoolAllocator<T> *allocator) const
    {
        static_assert(IsAsset<T>::value == true, "'T' is not of type Asset");

        return allocator != NULL ? (int)allocator->getCount() : 0;
    }

    template <typename T> T *getSystem_impl(const PoolAllocator<T> *allocator)
    {
        static_assert(IsSystem<T>::value == true, "'T' is not of type System");

        return allocator != NULL ? allocator->get(0) : NULL;
    }

    template <typename T> T *getComponent_impl(const PoolAllocator<T> *allocator, const Guid &entityId)
    {
        static_assert(IsComponent<T>::value == true, "'T' is not of type Component");

        if (allocator == NULL)
        {
            return NULL;
        }

        auto it1 = mEntityIdToComponentIds.find(entityId);

        if (it1 != mEntityIdToComponentIds.end())
        {
            std::vector<std::pair<Guid, int>> &componentsOnEntity = it1->second;

            for (size_t i = 0; i < componentsOnEntity.size(); i++)
            {
                if (ComponentType<T>::type == componentsOnEntity[i].second)
                {

                    std::unordered_map<Guid, int>::const_iterator it2 =
                        mIdToGlobalIndex.find(componentsOnEntity[i].first);
                    if (it2 != mIdToGlobalIndex.end())
                    {
                        return allocator->get(it2->second);
                    }

                    break;
                }
            }
        }

        return NULL;
    }

    template <typename T> T *addComponent_impl(PoolAllocator<T> *allocator, const Guid &entityId)
    {
        static_assert(IsComponent<T>::value == true, "'T' is not of type Component");

        if (getTypeOf(entityId) != EntityType<Entity>::type)
        {
            return NULL;
        }

        int componentGlobalIndex = (int)allocator->getCount();
        int componentType = ComponentType<T>::type;
        Guid componentId = Guid::newGuid();

        T *component = allocator->construct();

        if (component != NULL)
        {
            component->mEntityId = entityId;
            component->mComponentId = componentId;

            addIdToGlobalIndexMap_impl<T>(componentId, componentGlobalIndex, componentType);

            mEntityIdToComponentIds[entityId].push_back(std::make_pair(componentId, componentType));

            mComponentIdsMarkedCreated.push_back(make_triple(entityId, componentId, componentType));
        }

        return component;
    }

    template <typename T> T *addComponent_impl(PoolAllocator<T> *allocator, const std::vector<char> &data)
    {
        static_assert(IsComponent<T>::value == true, "'T' is not of type Component");

        int componentGlobalIndex = (int)allocator->getCount();
        int componentType = ComponentType<T>::type;

        T *component = allocator->construct(data);

        if (component == NULL || component->mComponentId == Guid::INVALID || component->mEntityId == Guid::INVALID)
        {
            return NULL;
        }

        addIdToGlobalIndexMap_impl<T>(component->mComponentId, componentGlobalIndex, componentType);

        mEntityIdToComponentIds[component->mEntityId].push_back(std::make_pair(component->mComponentId, componentType));

        mComponentIdsMarkedCreated.push_back(make_triple(component->mEntityId, component->mComponentId, componentType));

        return component;
    }

    template <typename T> T *addSystem_impl(PoolAllocator<T> *allocator, int order)
    {
        static_assert(IsSystem<T>::value == true, "'T' is not of type System");

        int systemGlobalIndex = (int)allocator->getCount();
        int systemType = SystemType<T>::type;
        Guid systemId = Guid::newGuid();

        T *system = allocator->construct();

        if (system != NULL)
        {
            system->mSystemId = systemId;

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
        static_assert(IsSystem<T>::value == true, "'T' is not of type System");

        return allocator != NULL ? allocator->get(index) : NULL;
    }

    template <typename T> T *getSystemById_impl(const PoolAllocator<T> *allocator, const Guid &systemId)
    {
        static_assert(IsSystem<T>::value == true, "'T' is not of type System");

        if (allocator == NULL || SystemType<T>::type != getTypeOf(systemId))
        {
            return NULL;
        }

        return getById_impl<T>(mIdToGlobalIndex, allocator, systemId);
    }

    template <typename T> T *getAssetByIndex_impl(const PoolAllocator<T> *allocator, int index)
    {
        static_assert(IsAsset<T>::value == true, "'T' is not of type Asset");

        return allocator != NULL ? allocator->get(index) : NULL;
    }

    template <typename T> T *getAssetById_impl(const PoolAllocator<T> *allocator, const Guid &assetId)
    {
        static_assert(IsAsset<T>::value == true, "'T' is not of type Asset");

        if (allocator == NULL || AssetType<T>::type != getTypeOf(assetId))
        {
            return NULL;
        }

        return getById_impl<T>(mIdToGlobalIndex, allocator, assetId);
    }

    template <typename T> T *getComponentByIndex_impl(const PoolAllocator<T> *allocator, int index)
    {
        static_assert(IsComponent<T>::value == true, "'T' is not of type Component");

        return allocator != NULL ? allocator->get(index) : NULL;
    }

    template <typename T> T *getComponentById_impl(const PoolAllocator<T> *allocator, const Guid &componentId)
    {
        static_assert(IsComponent<T>::value == true, "'T' is not of type Component");

        if (allocator == NULL || ComponentType<T>::type != getTypeOf(componentId))
        {
            return NULL;
        }

        return getById_impl<T>(mIdToGlobalIndex, allocator, componentId);
    }

    template <typename T> T *createAsset_impl(PoolAllocator<T> *allocator)
    {
        static_assert(IsAsset<T>::value == true, "'T' is not of type Asset");

        int index = (int)allocator->getCount();
        int type = AssetType<T>::type;
        Guid id = Guid::newGuid();

        T *asset = allocator->construct();

        if (asset != NULL)
        {
            asset->mAssetId = id;

            addIdToGlobalIndexMap_impl<T>(id, index, type);
        }

        return asset;
    }

    template <typename T> T *createAsset_impl(PoolAllocator<T> *allocator, const std::vector<char> &data)
    {
        static_assert(IsAsset<T>::value == true, "'T' is not of type Asset");

        int index = (int)allocator->getCount();
        int type = AssetType<T>::type;
        Guid id = IsAssetInternal<T>::value ? PhysicsEngine::ExtactInternalAssetId<T>(data)
                                            : PhysicsEngine::ExtactAssetId<T>(data);

        if (id == Guid::INVALID)
        {
            return NULL;
        }

        T *asset = allocator->construct(data);

        if (asset != NULL)
        {
            asset->mAssetId = id;

            addIdToGlobalIndexMap_impl<T>(id, index, type);
        }

        return asset;
    }

    template <typename T> PoolAllocator<T> *getComponentAllocator_impl()
    {
        static_assert(IsComponent<T>::value == true, "'T' is not of type Component");

        std::unordered_map<int, Allocator *>::iterator it = mComponentAllocatorMap.find(ComponentType<T>::type);
        if (it != mComponentAllocatorMap.end())
        {
            return static_cast<PoolAllocator<T> *>(it->second);
        }

        return NULL;
    }

    template <typename T> PoolAllocator<T> *getComponentOrAddAllocator_impl()
    {
        static_assert(IsComponent<T>::value == true, "'T' is not of type Component");

        PoolAllocator<T> *allocator = getComponentAllocator_impl<T>();
        if (allocator == NULL)
        {
            allocator = new PoolAllocator<T>();
            mComponentAllocatorMap[ComponentType<T>::type] = allocator;
        }

        return allocator;
    }

    template <typename T> PoolAllocator<T> *getSystemAllocator_impl()
    {
        static_assert(IsSystem<T>::value == true, "'T' is not of type System");

        std::unordered_map<int, Allocator *>::iterator it = mSystemAllocatorMap.find(SystemType<T>::type);
        if (it != mSystemAllocatorMap.end())
        {
            return static_cast<PoolAllocator<T> *>(it->second);
        }

        return NULL;
    }

    template <typename T> PoolAllocator<T> *getSystemOrAddAllocator_impl()
    {
        static_assert(IsSystem<T>::value == true, "'T' is not of type System");

        PoolAllocator<T> *allocator = getSystemAllocator_impl<T>();
        if (allocator == NULL)
        {
            allocator = new PoolAllocator<T>();
            mSystemAllocatorMap[SystemType<T>::type] = allocator;
        }

        return allocator;
    }

    template <typename T> PoolAllocator<T> *getAssetAllocator_impl()
    {
        static_assert(IsAsset<T>::value == true, "'T' is not of type Asset");

        std::unordered_map<int, Allocator *>::iterator it = mAssetAllocatorMap.find(AssetType<T>::type);
        if (it != mAssetAllocatorMap.end())
        {
            return static_cast<PoolAllocator<T> *>(it->second);
        }

        return NULL;
    }

    template <typename T> PoolAllocator<T> *getAssetOrAddAllocator_impl()
    {
        static_assert(IsAsset<T>::value == true, "'T' is not of type Asset");

        PoolAllocator<T> *allocator = getAssetAllocator_impl<T>();
        if (allocator == NULL)
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
            return NULL;
        }
    }

    template <typename T> void addIdToGlobalIndexMap_impl(const Guid &id, int index, int type)
    {
        mIdToGlobalIndex[id] = index;
        mIdToType[id] = type;
    }

    template <int N> void removeIdToGlobalIndexMap_impl(const Guid &id, int type)
    {
        mIdToGlobalIndex.erase(id);
        mIdToType.erase(id);
    }

    template <typename T> bool verifyData_impl(const std::vector<T> &data) const
    {
        return true;
    }

    // Explicit specializations

    template <>
    Transform *getComponentById_impl<Transform>(const PoolAllocator<Transform> *allocator, const Guid &componentId)
    {
        return getById_impl<Transform>(mTransformIdToGlobalIndex, allocator, componentId);
    }

    template <>
    MeshRenderer *getComponentById_impl<MeshRenderer>(const PoolAllocator<MeshRenderer> *allocator,
                                                      const Guid &componentId)
    {
        return getById_impl<MeshRenderer>(mMeshRendererIdToGlobalIndex, allocator, componentId);
    }

    template <>
    LineRenderer *getComponentById_impl<LineRenderer>(const PoolAllocator<LineRenderer> *allocator,
                                                      const Guid &componentId)
    {
        return getById_impl<LineRenderer>(mLineRendererIdToGlobalIndex, allocator, componentId);
    }

    template <>
    Rigidbody *getComponentById_impl<Rigidbody>(const PoolAllocator<Rigidbody> *allocator, const Guid &componentId)
    {
        return getById_impl<Rigidbody>(mRigidbodyIdToGlobalIndex, allocator, componentId);
    }

    template <> Camera *getComponentById_impl<Camera>(const PoolAllocator<Camera> *allocator, const Guid &componentId)
    {
        return getById_impl<Camera>(mCameraIdToGlobalIndex, allocator, componentId);
    }

    template <> Light *getComponentById_impl<Light>(const PoolAllocator<Light> *allocator, const Guid &componentId)
    {
        return getById_impl<Light>(mLightIdToGlobalIndex, allocator, componentId);
    }

    template <>
    SphereCollider *getComponentById_impl<SphereCollider>(const PoolAllocator<SphereCollider> *allocator,
                                                          const Guid &componentId)
    {
        return getById_impl<SphereCollider>(mSphereColliderIdToGlobalIndex, allocator, componentId);
    }

    template <>
    BoxCollider *getComponentById_impl<BoxCollider>(const PoolAllocator<BoxCollider> *allocator,
                                                    const Guid &componentId)
    {
        return getById_impl<BoxCollider>(mBoxColliderIdToGlobalIndex, allocator, componentId);
    }

    template <>
    CapsuleCollider *getComponentById_impl<CapsuleCollider>(const PoolAllocator<CapsuleCollider> *allocator,
                                                            const Guid &componentId)
    {
        return getById_impl<CapsuleCollider>(mCapsuleColliderIdToGlobalIndex, allocator, componentId);
    }

    template <>
    MeshCollider *getComponentById_impl<MeshCollider>(const PoolAllocator<MeshCollider> *allocator,
                                                      const Guid &componentId)
    {
        return getById_impl<MeshCollider>(mMeshColliderIdToGlobalIndex, allocator, componentId);
    }

    template <> Mesh *getAssetById_impl<Mesh>(const PoolAllocator<Mesh> *allocator, const Guid &assetId)
    {
        return getById_impl<Mesh>(mMeshIdToGlobalIndex, allocator, assetId);
    }

    template <> Material *getAssetById_impl<Material>(const PoolAllocator<Material> *allocator, const Guid &assetId)
    {
        return getById_impl<Material>(mMaterialIdToGlobalIndex, allocator, assetId);
    }

    template <> Shader *getAssetById_impl<Shader>(const PoolAllocator<Shader> *allocator, const Guid &assetId)
    {
        return getById_impl<Shader>(mShaderIdToGlobalIndex, allocator, assetId);
    }

    template <> Texture2D *getAssetById_impl<Texture2D>(const PoolAllocator<Texture2D> *allocator, const Guid &assetId)
    {
        return getById_impl<Texture2D>(mTexture2DIdToGlobalIndex, allocator, assetId);
    }

    template <> Texture3D *getAssetById_impl<Texture3D>(const PoolAllocator<Texture3D> *allocator, const Guid &assetId)
    {
        return getById_impl<Texture3D>(mTexture3DIdToGlobalIndex, allocator, assetId);
    }

    template <> Cubemap *getAssetById_impl<Cubemap>(const PoolAllocator<Cubemap> *allocator, const Guid &assetId)
    {
        return getById_impl<Cubemap>(mCubemapIdToGlobalIndex, allocator, assetId);
    }

    template <> Font *getAssetById_impl<Font>(const PoolAllocator<Font> *allocator, const Guid &assetId)
    {
        return getById_impl<Font>(mFontIdToGlobalIndex, allocator, assetId);
    }

    template <>
    RenderSystem *getSystemById_impl<RenderSystem>(const PoolAllocator<RenderSystem> *allocator, const Guid &assetId)
    {
        return getById_impl<RenderSystem>(mRenderSystemIdToGlobalIndex, allocator, assetId);
    }

    template <>
    PhysicsSystem *getSystemById_impl<PhysicsSystem>(const PoolAllocator<PhysicsSystem> *allocator, const Guid &assetId)
    {
        return getById_impl<PhysicsSystem>(mPhysicsSystemIdToGlobalIndex, allocator, assetId);
    }

    template <>
    CleanUpSystem *getSystemById_impl<CleanUpSystem>(const PoolAllocator<CleanUpSystem> *allocator, const Guid &assetId)
    {
        return getById_impl<CleanUpSystem>(mCleanupSystemIdToGlobalIndex, allocator, assetId);
    }

    template <>
    DebugSystem *getSystemById_impl<DebugSystem>(const PoolAllocator<DebugSystem> *allocator, const Guid &assetId)
    {
        return getById_impl<DebugSystem>(mDebugSystemIdToGlobalIndex, allocator, assetId);
    }
    template <>
    GizmoSystem* getSystemById_impl<GizmoSystem>(const PoolAllocator<GizmoSystem>* allocator, const Guid& assetId)
    {
        return getById_impl<GizmoSystem>(mGizmoSystemIdToGlobalIndex, allocator, assetId);
    }

    template <> void addIdToGlobalIndexMap_impl<Entity>(const Guid &id, int index, int type)
    {
        mEntityIdToGlobalIndex[id] = index;
        mIdToGlobalIndex[id] = index;
        mIdToType[id] = type;
    }

    template <> void addIdToGlobalIndexMap_impl<Transform>(const Guid &id, int index, int type)
    {
        mTransformIdToGlobalIndex[id] = index;
        mIdToGlobalIndex[id] = index;
        mIdToType[id] = type;
    }

    template <> void addIdToGlobalIndexMap_impl<MeshRenderer>(const Guid &id, int index, int type)
    {
        mMeshRendererIdToGlobalIndex[id] = index;
        mIdToGlobalIndex[id] = index;
        mIdToType[id] = type;
    }

    template <> void addIdToGlobalIndexMap_impl<LineRenderer>(const Guid &id, int index, int type)
    {
        mLineRendererIdToGlobalIndex[id] = index;
        mIdToGlobalIndex[id] = index;
        mIdToType[id] = type;
    }

    template <> void addIdToGlobalIndexMap_impl<Rigidbody>(const Guid &id, int index, int type)
    {
        mRigidbodyIdToGlobalIndex[id] = index;
        mIdToGlobalIndex[id] = index;
        mIdToType[id] = type;
    }

    template <> void addIdToGlobalIndexMap_impl<Camera>(const Guid &id, int index, int type)
    {
        mCameraIdToGlobalIndex[id] = index;
        mIdToGlobalIndex[id] = index;
        mIdToType[id] = type;
    }

    template <> void addIdToGlobalIndexMap_impl<Light>(const Guid &id, int index, int type)
    {
        mLightIdToGlobalIndex[id] = index;
        mIdToGlobalIndex[id] = index;
        mIdToType[id] = type;
    }

    template <> void addIdToGlobalIndexMap_impl<SphereCollider>(const Guid &id, int index, int type)
    {
        mSphereColliderIdToGlobalIndex[id] = index;
        mIdToGlobalIndex[id] = index;
        mIdToType[id] = type;
    }

    template <> void addIdToGlobalIndexMap_impl<BoxCollider>(const Guid &id, int index, int type)
    {
        mBoxColliderIdToGlobalIndex[id] = index;
        mIdToGlobalIndex[id] = index;
        mIdToType[id] = type;
    }

    template <> void addIdToGlobalIndexMap_impl<CapsuleCollider>(const Guid &id, int index, int type)
    {
        mCapsuleColliderIdToGlobalIndex[id] = index;
        mIdToGlobalIndex[id] = index;
        mIdToType[id] = type;
    }

    template <> void addIdToGlobalIndexMap_impl<MeshCollider>(const Guid &id, int index, int type)
    {
        mMeshColliderIdToGlobalIndex[id] = index;
        mIdToGlobalIndex[id] = index;
        mIdToType[id] = type;
    }

    template <> void addIdToGlobalIndexMap_impl<Mesh>(const Guid &id, int index, int type)
    {
        mMeshIdToGlobalIndex[id] = index;
        mIdToGlobalIndex[id] = index;
        mIdToType[id] = type;
    }

    template <> void addIdToGlobalIndexMap_impl<Material>(const Guid &id, int index, int type)
    {
        mMaterialIdToGlobalIndex[id] = index;
        mIdToGlobalIndex[id] = index;
        mIdToType[id] = type;
    }

    template <> void addIdToGlobalIndexMap_impl<Shader>(const Guid &id, int index, int type)
    {
        mShaderIdToGlobalIndex[id] = index;
        mIdToGlobalIndex[id] = index;
        mIdToType[id] = type;
    }

    template <> void addIdToGlobalIndexMap_impl<Texture2D>(const Guid &id, int index, int type)
    {
        mTexture2DIdToGlobalIndex[id] = index;
        mIdToGlobalIndex[id] = index;
        mIdToType[id] = type;
    }

    template <> void addIdToGlobalIndexMap_impl<Texture3D>(const Guid &id, int index, int type)
    {
        mTexture3DIdToGlobalIndex[id] = index;
        mIdToGlobalIndex[id] = index;
        mIdToType[id] = type;
    }

    template <> void addIdToGlobalIndexMap_impl<Cubemap>(const Guid &id, int index, int type)
    {
        mCubemapIdToGlobalIndex[id] = index;
        mIdToGlobalIndex[id] = index;
        mIdToType[id] = type;
    }

    template <> void addIdToGlobalIndexMap_impl<Font>(const Guid &id, int index, int type)
    {
        mFontIdToGlobalIndex[id] = index;
        mIdToGlobalIndex[id] = index;
        mIdToType[id] = type;
    }

    template <> void addIdToGlobalIndexMap_impl<RenderSystem>(const Guid &id, int index, int type)
    {
        mRenderSystemIdToGlobalIndex[id] = index;
        mIdToGlobalIndex[id] = index;
        mIdToType[id] = type;
    }

    template <> void addIdToGlobalIndexMap_impl<PhysicsSystem>(const Guid &id, int index, int type)
    {
        mPhysicsSystemIdToGlobalIndex[id] = index;
        mIdToGlobalIndex[id] = index;
        mIdToType[id] = type;
    }

    template <> void addIdToGlobalIndexMap_impl<CleanUpSystem>(const Guid &id, int index, int type)
    {
        mCleanupSystemIdToGlobalIndex[id] = index;
        mIdToGlobalIndex[id] = index;
        mIdToType[id] = type;
    }

    template <> void addIdToGlobalIndexMap_impl<DebugSystem>(const Guid &id, int index, int type)
    {
        mDebugSystemIdToGlobalIndex[id] = index;
        mIdToGlobalIndex[id] = index;
        mIdToType[id] = type;
    }

    template <> void addIdToGlobalIndexMap_impl<GizmoSystem>(const Guid& id, int index, int type)
    {
        mGizmoSystemIdToGlobalIndex[id] = index;
        mIdToGlobalIndex[id] = index;
        mIdToType[id] = type;
    }
};
} // namespace PhysicsEngine

#endif