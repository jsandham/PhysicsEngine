#ifndef RENDERSYSTEM_H__
#define RENDERSYSTEM_H__

#include <vector>

#include "System.h"

#include "../core/Input.h"

#include "../components/Camera.h"

#include "../graphics/DebugRenderer.h"
#include "../graphics/DeferredRenderer.h"
#include "../graphics/ForwardRenderer.h"
#include "../graphics/RenderObject.h"

namespace PhysicsEngine
{
class RenderSystem : public System
{
  private:
    ForwardRenderer mForwardRenderer;
    DeferredRenderer mDeferredRenderer;
    DebugRenderer mDebugRenderer;

    std::vector<glm::mat4> mTotalModels;
    std::vector<Id> mTotalTransformIds;
    std::vector<Sphere> mTotalBoundingSpheres;
    std::vector<RenderObject> mTotalRenderObjects;
    
    std::vector<glm::mat4> mModels;
    std::vector<Id> mTransformIds;
    std::vector<RenderObject> mRenderObjects;

    //std::vector<bool> mCulledObjectFlags;

  public:
    RenderSystem(World *world, const Id &id);
    RenderSystem(World *world, const Guid &guid, const Id &id);
    ~RenderSystem();

    virtual void serialize(YAML::Node &out) const override;
    virtual void deserialize(const YAML::Node &in) override;

    virtual int getType() const override;
    virtual std::string getObjectName() const override;

    void init(World *world) override;
    void update(const Input &input, const Time &time) override;

  private:
    void registerRenderAssets(World *world);
    void buildRenderObjectsList(World *world);
    void cullRenderObjects(Camera *camera);
    void buildRenderQueue();
    void sortRenderQueue();
    Sphere computeWorldSpaceBoundingSphere(const glm::mat4 &model, const Sphere &sphere);
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