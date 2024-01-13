#ifndef RENDERSYSTEM_H__
#define RENDERSYSTEM_H__

#include <vector>
#include <limits>

#include "../core/SerializationEnums.h"
#include "../core/Guid.h"
#include "../core/Id.h"
#include "../core/AABB.h"
#include "../core/Sphere.h"
#include "../core/BVH.h"

#include "../components/Camera.h"
#include "../components/Transform.h"

#include "../graphics/DebugRenderer.h"
#include "../graphics/DeferredRenderer.h"
#include "../graphics/ForwardRenderer.h"
#include "../graphics/RenderObject.h"
#include "../graphics/GraphicsQuery.h"
#include "../graphics/Raytracer.h"

namespace PhysicsEngine
{
class World;

struct Batch //BatchDrawCall?
{
    VertexBuffer *mVertexBuffer;
    VertexBuffer *mNormalBuffer;
    VertexBuffer *mTexCoordsBuffer;
    IndexBuffer *mIndexBuffer;
    MeshHandle *mMeshHandle;
};

class RenderSystem
{
  private:
    Guid mGuid;
    Id mId;
    World* mWorld;

    // Renderers
    ForwardRenderer mForwardRenderer;
    DeferredRenderer mDeferredRenderer;
    DebugRenderer mDebugRenderer;
    Raytracer mRaytracer;

    // Cache data that is used in contructing draw call lists once
    std::vector<glm::mat4> mCachedModels;
    std::vector<Id> mCachedTransformIds;
    std::vector<Sphere> mCachedBoundingSpheres;
    std::vector<AABB> mCachedBoundingAABBs;
    std::vector<int> mCachedMeshIndices;
    std::vector<int> mCachedMaterialIndices;
    AABB mCachedBoundingVolume;

    // Bounding Volume Hierarchy of scene
    BVH mBVH;

    // Frustum culling
    std::vector<int> mFrustumVisible;
    std::vector<std::pair<float, int>> mDistanceToCamera;

    // Occlusion culling
    VertexBuffer *mOcclusionVertexBuffer;
    VertexBuffer *mOcclusionModelIndexBuffer;
    IndexBuffer *mOcclusionIndexBuffer;
    MeshHandle *mOcclusionMeshHandle;
    OcclusionMapShader *mOcclusionMapShader;
    OcclusionUniform *mOcclusionModelMatUniform;
    std::vector<float> mOccluderVertices;
    std::vector<int> mOccluderModelIndices;
    std::vector<unsigned int> mOccluderIndices;
    OcclusionQuery* mOcclusionQuery[2];
    int mOcclusionQueryIndex;

    // RenderQueue
    std::vector<std::pair<DrawCallCommand, int>> mRenderQueueScratch;
    std::vector<std::pair<DrawCallCommand, int>> mRenderQueue;

    // Draw call data
    std::vector<glm::mat4> mModels;
    std::vector<Id> mTransformIds;
    std::vector<Sphere> mBoundingSpheres;

    std::vector<DrawCallCommand> mDrawCallCommands;
    std::vector<Batch> mBatches;

  public:
    HideFlag mHide;
    bool mEnabled;
    bool mRaytraceEnabled;

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
    void update();

    const BVH &getBVH() const;

  private:
    void registerRenderAssets();
    void cacheRenderData();
    void allocateBVH();
    void freeBVH();
    void buildBVH();
    void frustumCulling(const Camera *camera);
    void occlusionCulling(const Camera *camera);
    void buildRenderQueue();
    void sortRenderQueue();
    void buildDrawCallCommandList();
    Sphere computeWorldSpaceBoundingSphere(const glm::mat4 &model, const Sphere &sphere);
};

} // namespace PhysicsEngine

#endif