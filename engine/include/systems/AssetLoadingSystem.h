#ifndef ASSET_LOADING_SYSTEM_H__
#define ASSET_LOADING_SYSTEM_H__

#include <queue>
#include <thread>
#include <vector>

#include "../core/SerializationEnums.h"
#include "../core/AssetEnums.h"
#include "../core/Guid.h"
#include "../core/Id.h"
#include "../core/Input.h"
#include "../core/Time.h"

namespace PhysicsEngine
{
struct Texture2DLoadingRequest
{
    std::vector<unsigned char> mData;
    std::string mFilepath;
    int mWidth;
    int mHeight;
    TextureFormat mFormat;
    std::atomic<bool> mIdle;
    std::atomic<bool> mDone;
};

class World;

class AssetLoadingSystem
{
  private:
    Guid mGuid;
    Id mId;
    World* mWorld;

    std::queue<std::string> mTextureQueue;
    // std::vector<Texture2DLoadingRequest> mWorkerRequests;
    // std::vector<std::thread> mWorkers;
    // unsigned int mNumThreads;
    Texture2DLoadingRequest mWorkerRequests[12];
    std::thread mWorkers[12];
    unsigned int mNumThreads;

  public:
    HideFlag mHide;
    bool mEnabled;

  public:
    AssetLoadingSystem(World *world, const Id &id);
    AssetLoadingSystem(World *world, const Guid &guid, const Id &id);
    ~AssetLoadingSystem();

    void serialize(YAML::Node &out) const;
    void deserialize(const YAML::Node &in);

    int getType() const;
    std::string getObjectName() const;

    Guid getGuid() const;
    Id getId() const;

    void init(World *world);
    void update(const Input &input, const Time &time);

    void loadTexture2DAsync(const std::string &filepath);

    void doWork(int slot);
};

} // namespace PhysicsEngine

#endif