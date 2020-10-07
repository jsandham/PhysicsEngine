#ifndef __RENDERSYSTEM_H__
#define __RENDERSYSTEM_H__

#include <vector>

#include "System.h"

#include "../core/Input.h"

#include "../components/Camera.h"

#include "../graphics/DeferredRenderer.h"
#include "../graphics/ForwardRenderer.h"
#include "../graphics/RenderObject.h"

namespace PhysicsEngine
{
#pragma pack(push, 1)
struct RenderSystemHeader
{
    Guid mSystemId;
    int32_t mUpdateOrder;
};
#pragma pack(pop)

class RenderSystem : public System
{
  private:
    ForwardRenderer mForwardRenderer;
    DeferredRenderer mDeferredRenderer;

    std::vector<RenderObject> mRenderObjects;
    std::vector<std::pair<uint64_t, int>> mRenderQueue;

  public:
    bool mRenderToScreen;

  public:
    RenderSystem();
    RenderSystem(const std::vector<char> &data);
    ~RenderSystem();

    std::vector<char> serialize() const;
    std::vector<char> serialize(Guid systemId) const;
    void deserialize(const std::vector<char> &data);

    void init(World *world);
    void update(Input input, Time time);

  private:
    void registerRenderAssets(World *world);
    void registerCameras(World *world);
    void registerLights(World *world);
    void buildRenderObjectsList(World *world);
    void cullRenderObjects(Camera *camera);
    void buildRenderQueue();
    void sortRenderQueue();

    void updateRenderObjects(World *world);
};

template <typename T> struct IsRenderSystem
{
    static constexpr bool value = false;
};

template <> struct SystemType<RenderSystem>
{
    static constexpr int type = PhysicsEngine::RENDERSYSTEM_TYPE;
};
template <> struct IsRenderSystem<RenderSystem>
{
    static constexpr bool value = true;
};
template <> struct IsSystem<RenderSystem>
{
    static constexpr bool value = true;
};
template <> struct IsSystemInternal<RenderSystem>
{
    static constexpr bool value = true;
};
} // namespace PhysicsEngine

#endif