#ifndef RENDERSYSTEM_H__
#define RENDERSYSTEM_H__

#include <vector>

#include "../core/SerializationEnums.h"
#include "../core/Guid.h"
#include "../core/Id.h"
#include "../core/Input.h"
#include "../core/Time.h"

#include "../components/Camera.h"

#include "../graphics/DebugRenderer.h"
#include "../graphics/DeferredRenderer.h"
#include "../graphics/ForwardRenderer.h"
#include "../graphics/RenderObject.h"

namespace PhysicsEngine
{
class World;

class RenderSystem
{
  private:
    Guid mGuid;
    Id mId;
    World* mWorld;

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

    // std::vector<bool> mCulledObjectFlags;

  public:
    HideFlag mHide;

  public:
    RenderSystem(World *world, const Id &id);
    RenderSystem(World *world, const Guid &guid, const Id &id);
    ~RenderSystem();

    void serialize(YAML::Node &out) const;
    void deserialize(const YAML::Node &in);

    int getType() const;
    std::string getObjectName() const;

    Guid getGuid() const;
    Id getId() const;

    void init(World *world);
    void update(const Input &input, const Time &time);

  private:
    void registerRenderAssets(World *world);
    void buildRenderObjectsList(World *world);
    void cullRenderObjects(Camera *camera);
    void buildRenderQueue();
    void sortRenderQueue();
    Sphere computeWorldSpaceBoundingSphere(const glm::mat4 &model, const Sphere &sphere);
};

} // namespace PhysicsEngine

#endif