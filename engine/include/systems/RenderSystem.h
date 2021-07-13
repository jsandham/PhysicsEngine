#ifndef RENDERSYSTEM_H__
#define RENDERSYSTEM_H__

#include <vector>

#include "System.h"

#include "../core/Input.h"

#include "../components/Camera.h"

#include "../graphics/DeferredRenderer.h"
#include "../graphics/ForwardRenderer.h"
#include "../graphics/RenderObject.h"
#include "../graphics/SpriteObject.h"

namespace PhysicsEngine
{
class RenderSystem : public System
{
  private:
    ForwardRenderer mForwardRenderer;
    DeferredRenderer mDeferredRenderer;

    std::vector<RenderObject> mRenderObjects;
    std::vector<SpriteObject> mSpriteObjects;
    std::vector<std::pair<uint64_t, int>> mRenderQueue;

  public:
    bool mRenderToScreen;

  public:
    RenderSystem();
    RenderSystem(Guid id);
    ~RenderSystem();

    virtual void serialize(YAML::Node &out) const override;
    virtual void deserialize(const YAML::Node &in) override;

    virtual int getType() const override;
    virtual std::string getObjectName() const override;

    void init(World *world) override;
    void update(const Input &input, const Time &time) override;

  private:
    void registerRenderAssets(World *world);
    void registerCameras(World *world);
    void registerLights(World *world);
    void buildRenderObjectsList(World *world);
    void buildSpriteObjectsList(World* world);
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