#ifndef WORLD_H__
#define WORLD_H__

#include <filesystem>

#define GLM_FORCE_RADIANS

#include "WorldAllocators.h"
#include "WorldIdState.h"
#include "WorldPrimitives.h"

namespace PhysicsEngine
{
class World
{
  private:
    // allocators for scenes, assets, and systems
    WorldAllocators mAllocators;

    // id state for scenes, assets, and systems
    WorldIdState mIdState;

    // Primitive meshes all worlds have access to
    WorldPrimitives mPrimitives;

    // active scene
    Scene *mActiveScene;

    // all systems in world listed in order they should be updated
    std::vector<System *> mSystems;

    std::unordered_map<Guid, std::unordered_map<Guid, std::vector<ShaderUniform>>> mMaterialUniformCache;

  public:
    std::vector<Sphere> mBoundingSpheres;

  public:
    World();
    ~World();
    World(const World &other) = delete;
    World &operator=(const World &other) = delete;

    void loadAssetsInPath(const std::filesystem::path &filePath);
    Asset *loadAssetFromYAML(const std::string &filePath);
    Scene *loadSceneFromYAML(const std::string &filePath);
    bool writeAssetToYAML(const std::string &filePath, const Guid &assetGuid) const;
    bool writeSceneToYAML(const std::string &filePath, const Guid &sceneGuid) const;

    void copyDoNotDestroyEntities(Scene* from, Scene* to);

    std::vector<ShaderUniform> getCachedMaterialUniforms(const Guid &materialGuid, const Guid &shaderGuid);
    void cacheMaterialUniforms(const Guid &materialGuid, const Guid &shaderGuid, const std::vector<ShaderUniform>& uniforms);

    size_t getNumberOfScenes() const;
    size_t getNumberOfUpdatingSystems() const;
    Mesh *getPrimtiveMesh(PrimitiveType type) const;
    Material *getPrimtiveMaterial() const;

    Scene *getActiveScene();

    Asset *getAssetById(const Id &assetId, int type) const;
    Asset *getAssetByGuid(const Guid &assetGuid, int type) const;
    Scene *getSceneByIndex(size_t index) const;
    Scene *getSceneById(const Id &sceneId) const;
    Scene *getSceneByGuid(const Guid &sceneGuid) const;
    System *getSystemByUpdateOrder(size_t order) const;

    Asset *createAsset(const YAML::Node &in, int type);

    int getIndexOf(const Id &id) const;
    int getIndexOf(const Guid &guid) const;
    int getTypeOf(const Id &id) const;
    int getTypeOf(const Guid &guid) const;

    Scene *createScene();
    Scene *createScene(const YAML::Node &in);
    void latentDestroyAsset(const Guid &assetGuid, int assetType);
    void immediateDestroyAsset(const Guid &assetGuid, int assetType);

    template <typename T> size_t getNumberOfSystems() const;
    template <typename T> size_t getNumberOfAssets() const;
    template <typename T> T* getSystem() const;
    template <typename T> T* addSystem(size_t order);
    template <typename T> T* getSystemByIndex(size_t index) const;
    template <typename T> T* getSystemById(const Id& systemId) const;
    template <typename T> T* getSystemByGuid(const Guid &systemGuid) const;
    template <typename T> T* getAssetByIndex(size_t index) const;
    template <typename T> T* getAssetById(const Id& assetId) const;
    template <typename T> T* getAssetByGuid(const Guid &assetGuid) const;
    template <typename T> T* createAsset();
    template <typename T> T* createAsset(const Guid& assetGuid);
    template <typename T> T* createAsset(const YAML::Node& in);

  private:
    void generateSourcePaths(const std::string& filepath, YAML::Node &in);
    void addToIdState(const Guid &guid, const Id &id, int index, int type);
    void removeFromIdState(const Guid &guid, const Id &id);

    template <typename T> void addToIdState_impl(const Guid &guid, const Id &id, int index, int type);
    template <typename T> void removeFromIdState_impl(const Guid &guid, const Id &id);
    template <typename T> T* addSystem_impl(PoolAllocator<T>* allocator, size_t order);
    template <typename T> T* getSystemById_impl(const std::unordered_map<Id, int>& idToIndexMap, const PoolAllocator<T>* allocator, const Id& systemId) const;
    template <typename T> T* getSystemByGuid_impl(const std::unordered_map<Guid, int> &guidToIndexMap, const PoolAllocator<T> *allocator, const Guid &systemGuid) const;
    template <typename T> T* getAssetById_impl(const std::unordered_map<Id, int>& idToIndexMap, const PoolAllocator<T>* allocator, const Id& assetId) const;
    template <typename T> T* getAssetByGuid_impl(const std::unordered_map<Guid, int> &guidToIndexMap, const PoolAllocator<T> *allocator, const Guid &assetGuid) const;
    template <typename T> T *createAsset_impl(PoolAllocator<T> *allocator, const Guid &assetGuid);
    template <typename T> T* createAsset_impl(PoolAllocator<T>* allocator, const YAML::Node& in);
    template <typename T> T* getById_impl(const std::unordered_map<Id, int> &idToIndexMap, const PoolAllocator<T> *allocator, const Id &id) const;
    template <typename T> T *getByGuid_impl(const std::unordered_map<Guid, int> &guidToIndexMap, const PoolAllocator<T> *allocator, const Guid &guid) const;
};
} // namespace PhysicsEngine

#endif