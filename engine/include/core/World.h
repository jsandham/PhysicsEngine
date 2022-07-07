#ifndef WORLD_H__
#define WORLD_H__

#include <string>
#include <unordered_map>
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
    bool writeAssetToYAML(const std::string &filePath, const Guid &assetId) const;
    bool writeSceneToYAML(const std::string &filePath, const Guid &sceneId) const;

    void copyDoNotDestroyEntities(Scene* from, Scene* to);

    std::vector<ShaderUniform> getCachedMaterialUniforms(const Guid &materialId, const Guid &shaderId);
    void cacheMaterialUniforms(const Guid &materialId, const Guid &shaderId, const std::vector<ShaderUniform>& uniforms);

    size_t getNumberOfScenes() const;
    size_t getNumberOfUpdatingSystems() const;
    Mesh *getPrimtiveMesh(PrimitiveType type) const;
    Material *World::getPrimtiveMaterial() const;

    Scene *getActiveScene();

    Asset *getAssetById(const Guid &assetId, int type) const;
    Scene *getSceneByIndex(size_t index) const;
    Scene *getSceneById(const Guid &sceneId) const;
    System *getSystemByUpdateOrder(size_t order) const;

    Asset *createAsset(const YAML::Node &in, int type);

    int getIndexOf(const Guid &id) const;
    int getTypeOf(const Guid &id) const;

    Scene *createScene();
    Scene *World::createScene(const YAML::Node &in);
    void latentDestroyAsset(const Guid &assetId, int assetType);
    void immediateDestroyAsset(const Guid &assetId, int assetType);

    std::string getAssetFilepath(const Guid &assetId) const;
    std::string getSceneFilepath(const Guid &sceneId) const;

    Guid getAssetId(const std::string &filepath) const;
    Guid getSceneId(const std::string &filepath) const;

    template <typename T> size_t getNumberOfSystems() const;
    template <typename T> size_t getNumberOfAssets() const;
    template <typename T> T* getSystem() const;
    template <typename T> T* addSystem(size_t order);
    template <typename T> T* getSystemByIndex(size_t index) const;
    template <typename T> T* getSystemById(const Guid& systemId) const;
    template <typename T> T* getAssetByIndex(size_t index) const;
    template <typename T> T* getAssetById(const Guid& assetId) const;
    template <typename T> T* createAsset();
    template <typename T> T* createAsset(const Guid& id);
    template <typename T> T* createAsset(const YAML::Node& in);

  private:
    void generateSourcePaths(const std::string& filepath, YAML::Node &in);

    template <typename T> void addIdToGlobalIndexMap_impl(const Guid& id, int index, int type);
    template <typename T> T* addSystem_impl(PoolAllocator<T>* allocator, size_t order);
    template <typename T> T* getSystemById_impl(const std::unordered_map<Guid, int>& idToIndexMap, const PoolAllocator<T>* allocator, const Guid& systemId) const;
    template <typename T> T* getAssetById_impl(const std::unordered_map<Guid, int>& idToIndexMap, const PoolAllocator<T>* allocator, const Guid& assetId) const;
    template <typename T> T* createAsset_impl(PoolAllocator<T>* allocator, const Guid& assetId);
    template <typename T> T* createAsset_impl(PoolAllocator<T>* allocator, const YAML::Node& in);
    template <typename T> T* getById_impl(const std::unordered_map<Guid, int>& idToIndexMap, const PoolAllocator<T>* allocator, const Guid& id) const;
};
} // namespace PhysicsEngine

#endif