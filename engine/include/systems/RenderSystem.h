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
    RenderSystem(Guid id);
    ~RenderSystem();

    std::vector<char> serialize() const;
    std::vector<char> serialize(const Guid &systemId) const;
    void deserialize(const std::vector<char> &data);

    void serialize(std::ostream& out) const;
    void deserialize(std::istream& in);

    void init(World *world);
    void update(const Input &input, const Time &time);

  private:
    void registerRenderAssets(World *world);
    void registerCameras(World *world);
    void registerLights(World *world);
    void buildRenderObjectsList(World *world);
    void cullRenderObjects(Camera *camera);
    void buildRenderQueue();
    void sortRenderQueue();
};

template <> struct SystemType<RenderSystem>
{
    static constexpr int type = PhysicsEngine::RENDERSYSTEM_TYPE;
};
template <> struct IsSystemInternal<RenderSystem>
{
    static constexpr bool value = true;
};
} // namespace PhysicsEngine

#endif