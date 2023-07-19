#ifndef ASSET_LOADING_SYSTEM_H__
#define ASSET_LOADING_SYSTEM_H__

#include <queue>
#include <thread>
#include <vector>

#include "../core/Texture.h"
#include "System.h"

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

class AssetLoadingSystem : public System
{
  private:
    std::queue<std::string> mTextureQueue;
    // std::vector<Texture2DLoadingRequest> mWorkerRequests;
    // std::vector<std::thread> mWorkers;
    // unsigned int mNumThreads;
    Texture2DLoadingRequest mWorkerRequests[12];
    std::thread mWorkers[12];
    unsigned int mNumThreads;

  public:
    AssetLoadingSystem(World *world, const Id &id);
    AssetLoadingSystem(World *world, const Guid &guid, const Id &id);
    ~AssetLoadingSystem();

    virtual void serialize(YAML::Node &out) const override;
    virtual void deserialize(const YAML::Node &in) override;

    virtual int getType() const override;
    virtual std::string getObjectName() const override;

    void init(World *world) override;
    void update(const Input &input, const Time &time) override;

    void loadTexture2DAsync(const std::string &filepath);

    void doWork(int slot);
};

template <> struct SystemType<AssetLoadingSystem>
{
    static constexpr int type = PhysicsEngine::ASSETLOADINGSYSTEM_TYPE;
};
template <> struct IsSystemInternal<AssetLoadingSystem>
{
    static constexpr bool value = true;
};
} // namespace PhysicsEngine

#endif