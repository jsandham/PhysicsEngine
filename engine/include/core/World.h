#ifndef WORLD_H__
#define WORLD_H__

#include <filesystem>

#define GLM_FORCE_RADIANS

#include "Guid.h"
#include "PoolAllocator.h"

#include "../core/Scene.h"

#include "../core/Cubemap.h"
#include "../core/Material.h"
#include "../core/Mesh.h"
#include "../core/Texture2D.h"

#include "../systems/AssetLoadingSystem.h"
#include "../systems/CleanUpSystem.h"
#include "../systems/DebugSystem.h"
#include "../systems/FreeLookCameraSystem.h"
#include "../systems/GizmoSystem.h"
#include "../systems/PhysicsSystem.h"
#include "../systems/RenderSystem.h"
#include "../systems/TerrainSystem.h"

#include "WorldPrimitives.h"

namespace PhysicsEngine
{
// Simple structs used for grouping world id to global index maps when passing to functions
struct WorldIdState
{
    // world scene guid state
    std::unordered_map<Guid, int> mSceneGuidToGlobalIndex;

    // world asset guid state
    std::unordered_map<Guid, int> mMeshGuidToGlobalIndex;
    std::unordered_map<Guid, int> mMaterialGuidToGlobalIndex;
    std::unordered_map<Guid, int> mShaderGuidToGlobalIndex;
    std::unordered_map<Guid, int> mTexture2DGuidToGlobalIndex;
    std::unordered_map<Guid, int> mTexture3DGuidToGlobalIndex;
    std::unordered_map<Guid, int> mCubemapGuidToGlobalIndex;
    std::unordered_map<Guid, int> mRenderTextureGuidToGlobalIndex;

    // world id state for all scenes, and assets
    std::unordered_map<Guid, int> mGuidToGlobalIndex;
    std::unordered_map<Guid, int> mGuidToType;

    // asset create/deletion state
    std::vector<std::pair<Guid, int>> mAssetGuidsMarkedCreated;
    std::vector<std::pair<Guid, int>> mAssetGuidsMarkedLatentDestroy;
    std::vector<std::pair<Guid, int>> mAssetGuidsMarkedMoved;

    // world scene id state
    std::unordered_map<Id, int> mSceneIdToGlobalIndex;

    // world asset id state
    std::unordered_map<Id, int> mMeshIdToGlobalIndex;
    std::unordered_map<Id, int> mMaterialIdToGlobalIndex;
    std::unordered_map<Id, int> mShaderIdToGlobalIndex;
    std::unordered_map<Id, int> mTexture2DIdToGlobalIndex;
    std::unordered_map<Id, int> mTexture3DIdToGlobalIndex;
    std::unordered_map<Id, int> mCubemapIdToGlobalIndex;
    std::unordered_map<Id, int> mRenderTextureIdToGlobalIndex;

    // world id state for all scenes, and assets
    std::unordered_map<Id, int> mIdToGlobalIndex;
    std::unordered_map<Id, int> mIdToType;

    std::unordered_map<Guid, Id> mGuidToId;
    std::unordered_map<Id, Guid> mIdToGuid;
};

// Simple structs used for grouping world allocators when passing to functions
struct WorldAllocators
{
    // scene allocator
    PoolAllocator<Scene> mSceneAllocator;

    // asset allocators
    PoolAllocator<Mesh> mMeshAllocator;
    PoolAllocator<Material> mMaterialAllocator;
    PoolAllocator<Shader> mShaderAllocator;
    PoolAllocator<Texture2D> mTexture2DAllocator;
    PoolAllocator<Cubemap> mCubemapAllocator;
    PoolAllocator<RenderTexture> mRenderTextureAllocator;
};

class World
{
  private:
    // allocators for scenes and assets
    WorldAllocators mAllocators;

    // id state for scenes and assets
    WorldIdState mIdState;

    // Primitive meshes all worlds have access to
    WorldPrimitives mPrimitives;

    // active scene
    Scene *mActiveScene;

    AssetLoadingSystem* mAssetLoadingSystem;
    CleanUpSystem* mCleanUpSystem;
    DebugSystem* mDebugSystem;
    FreeLookCameraSystem* mFreeLookCameraSystem;
    GizmoSystem* mGizmoSystem;
    PhysicsSystem* mPhysicsSystem;
    RenderSystem* mRenderSystem;
    TerrainSystem* mTerrainSystem;

    std::unordered_map<Guid, std::unordered_map<Guid, std::vector<ShaderUniform>>> mMaterialUniformCache;

  public:
    std::vector<Sphere> mBoundingSpheres;

  public:
    World();
    ~World();
    World(const World &other) = delete;
    World &operator=(const World &other) = delete;

    void loadAllAssetsInPath(const std::filesystem::path &filePath);
    Cubemap* loadCubemapFromYAML(const std::string &filePath);
    Material* loadMaterialFromYAML(const std::string &filePath);
    Mesh* loadMeshFromYAML(const std::string &filePath);
    RenderTexture* loadRenderTextureFromYAML(const std::string &filePath);
    Shader* loadShaderFromYAML(const std::string &filePath);
    Texture2D* loadTexture2DFromYAML(const std::string &filePath);

    Scene *loadSceneFromYAML(const std::string &filePath);
    bool writeAssetToYAML(const std::string &filePath, const Guid &assetGuid) const;
    bool writeSceneToYAML(const std::string &filePath, const Guid &sceneGuid) const;

    void copyDoNotDestroyEntities(Scene *from, Scene *to);

    std::vector<ShaderUniform> getCachedMaterialUniforms(const Guid &materialGuid, const Guid &shaderGuid);
    void cacheMaterialUniforms(const Guid &materialGuid, const Guid &shaderGuid,
                               const std::vector<ShaderUniform> &uniforms);

    size_t getNumberOfScenes() const;
    Mesh *getPrimtiveMesh(PrimitiveType type) const;
    Material *getPrimtiveMaterial() const;

    Scene *getActiveScene();

    Cubemap* getCubemapById(const Id &assetId) const;
    Material* getMaterialById(const Id &assetId) const;
    Mesh* getMeshById(const Id &assetId) const;
    RenderTexture* getRenderTexutreById(const Id &assetId) const;
    Shader* getShaderById(const Id &assetId) const;
    Texture2D* getTexture2DById(const Id &assetId) const;

    Cubemap* getCubemapByGuid(const Guid &assetGuid) const;
    Material* getMaterialByGuid(const Guid &assetGuid) const;
    Mesh* getMeshByGuid(const Guid &assetGuid) const;
    RenderTexture* getRenderTextureByGuid(const Guid &assetGuid) const;
    Shader* getShaderByGuid(const Guid &assetGuid) const;
    Texture2D* getTexture2DByGuid(const Guid &assetGuid) const;

    Scene *getSceneByIndex(size_t index) const;
    Scene *getSceneById(const Id &sceneId) const;
    Scene *getSceneByGuid(const Guid &sceneGuid) const;

    Cubemap *createCubemap(const YAML::Node &in);
    Material *createMaterial(const YAML::Node &in);
    Mesh *createMesh(const YAML::Node &in);
    RenderTexture *createRenderTexture(const YAML::Node &in);
    Shader *createShader(const YAML::Node &in);
    Texture2D *createTexture2D(const YAML::Node &in);

    int getIndexOf(const Id &id) const;
    int getIndexOf(const Guid &guid) const;
    int getTypeOf(const Id &id) const;
    int getTypeOf(const Guid &guid) const;

    Scene *createScene();
    Scene *createScene(const YAML::Node &in);
    void latentDestroyAsset(const Guid &assetGuid, int assetType);
    void immediateDestroyAsset(const Guid &assetGuid, int assetType);

    template <typename T> size_t getNumberOfAssets() const;
    template <typename T> T *getSystem() const;
    template <typename T> T *getAssetByIndex(size_t index) const;
    template <typename T> T *getAssetById(const Id &assetId) const;
    template <typename T> T *getAssetByGuid(const Guid &assetGuid) const;
    template <typename T> T *createAsset();
    template <typename T> T *createAsset(const Guid &assetGuid);
    template <typename T> T *createAsset(const YAML::Node &in);

  private:
    bool loadAssetYAML(const std::string &filePath, YAML::Node &in, Guid& guid, int& type);
    void generateSourcePaths(const std::string &filepath, YAML::Node &in);
    void addToIdState(const Guid &guid, const Id &id, int index, int type);
    void removeFromIdState(const Guid &guid, const Id &id);






    template <typename T> void addToIdState_impl(const Guid &guid, const Id &id, int index, int type);
    template <typename T> void removeFromIdState_impl(const Guid &guid, const Id &id);
    template <typename T>
    T *getAssetById_impl(const std::unordered_map<Id, int> &idToIndexMap, const PoolAllocator<T> *allocator,
                         const Id &assetId) const;
    template <typename T>
    T *getAssetByGuid_impl(const std::unordered_map<Guid, int> &guidToIndexMap, const PoolAllocator<T> *allocator,
                           const Guid &assetGuid) const;
    template <typename T> T *createAsset_impl(PoolAllocator<T> *allocator, const Guid &assetGuid);
    template <typename T> T *createAsset_impl(PoolAllocator<T> *allocator, const YAML::Node &in);
    template <typename T>
    T *getById_impl(const std::unordered_map<Id, int> &idToIndexMap, const PoolAllocator<T> *allocator,
                    const Id &id) const;
    template <typename T>
    T *getByGuid_impl(const std::unordered_map<Guid, int> &guidToIndexMap, const PoolAllocator<T> *allocator,
                      const Guid &guid) const;
};
} // namespace PhysicsEngine

#endif