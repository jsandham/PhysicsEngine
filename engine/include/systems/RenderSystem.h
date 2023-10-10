#ifndef RENDERSYSTEM_H__
#define RENDERSYSTEM_H__

#include <vector>

#include "../core/SerializationEnums.h"
#include "../core/Guid.h"
#include "../core/Id.h"
#include "../core/Input.h"
#include "../core/Time.h"

#include "../components/Camera.h"
#include "../components/Transform.h"

#include "../graphics/DebugRenderer.h"
#include "../graphics/DeferredRenderer.h"
#include "../graphics/ForwardRenderer.h"
#include "../graphics/RenderObject.h"

namespace PhysicsEngine
{
class World;

struct DrawCallData
{
    std::vector<RenderObject> mDrawCalls;
    std::vector<glm::mat4> mModels;
    std::vector<Id> mTransformIds;
    std::vector<Sphere> mBoundingSpheres;

    // std::vector<RenderObject> mInstancedDrawCalls;
    // std::vector<glm::mat4> mInstancedModels;
    // std::vector<Id> mInstancedTransformIds;
    // std::vector<Sphere> mInstancedBoundingSpheres;
};

class RenderSystem
{
  private:
    Guid mGuid;
    Id mId;
    World* mWorld;

    ForwardRenderer mForwardRenderer;
    DeferredRenderer mDeferredRenderer;
    DebugRenderer mDebugRenderer;

    // Cache data that is used in contructing draw call lists once
    std::vector<TransformData> mCachedTransforms;
    std::vector<Id> mCachedTransformIds;
    std::vector<Sphere> mCachedBoundingSpheres;
    std::vector<int> mCachedMeshIndices;
    std::vector<int> mCachedMaterialIndices;

    // Scratch arrays
    std::vector<RenderObject> mDrawCallScratch;
    std::vector<int> mDrawCallMeshRendererIndices;

    // Draw call data
    std::vector<RenderObject> mDrawCalls;
    std::vector<glm::mat4> mModels;
    std::vector<Id> mTransformIds;
    std::vector<Sphere> mBoundingSpheres;

    //std::vector<RenderObject> mInstancedDrawCalls;
    //std::vector<glm::mat4> mInstancedModels;
    //std::vector<Id> mInstancedTransformIds;
    //std::vector<Sphere> mInstancedBoundingSpheres;









    //std::vector<glm::mat4> mTotalModels;
    //std::vector<Id> mTotalTransformIds;
    //std::vector<Sphere> mTotalBoundingSpheres;
    //std::vector<RenderObject> mTotalRenderObjects;

    //std::vector<glm::mat4> mFinalModels;
    //std::vector<Id> mFinalTransformIds;
    //std::vector<RenderObject> mFinalRenderObjects;

    // std::vector<bool> mCulledObjectFlags;

  public:
    HideFlag mHide;
    bool mEnabled;

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
    void cacheRenderData(World *world);
    void buildRenderObjectsList(World *world, Camera* camera);
    void cullRenderObjects(Camera *camera);
    void buildRenderQueue();
    void sortRenderQueue();
    Sphere computeWorldSpaceBoundingSphere(const glm::mat4 &model, const Sphere &sphere);
    Sphere computeWorldSpaceBoundingSphere(const glm::vec3 &translation, const glm::vec3 &scale, const Sphere &sphere);
    Sphere computeWorldSpaceBoundingSphere(const glm::mat4 &model, const glm::vec3 &scale, const Sphere &sphere);
};

} // namespace PhysicsEngine

#endif