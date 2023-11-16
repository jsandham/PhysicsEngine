#ifndef RENDERSYSTEM_H__
#define RENDERSYSTEM_H__

#include <vector>
#include <limits>

#include "../core/SerializationEnums.h"
#include "../core/Guid.h"
#include "../core/Id.h"
#include "../core/AABB.h"
#include "../core/Sphere.h"

#include "../components/Camera.h"
#include "../components/Transform.h"

#include "../graphics/DebugRenderer.h"
#include "../graphics/DeferredRenderer.h"
#include "../graphics/ForwardRenderer.h"
#include "../graphics/RenderObject.h"

namespace PhysicsEngine
{
class World;

struct BVHNode
{
    glm::vec3 mMin;
    glm::vec3 mMax;
    int mLeft;
    int mStartIndex;
    int mIndexCount;

    inline bool isLeaf() const
    {
        return mIndexCount > 0;
    }
};

struct BVH
{
    std::vector<BVHNode> mNodes;
    std::vector<int> mPerm;

    void buildBVH(const std::vector<AABB> &boundingAABBs)
    {
        if (boundingAABBs.size() == 0)
        {
            return;
        }

        mPerm.resize(boundingAABBs.size());
        for (size_t i = 0; i < mPerm.size(); i++)
        {
            mPerm[i] = (int)i;
        }

        mNodes.resize(2 * boundingAABBs.size() - 1);

        mNodes[0].mLeft = 0;
        mNodes[0].mStartIndex = 0;
        mNodes[0].mIndexCount = static_cast<int>(boundingAABBs.size());

        std::queue<int> queue;
        queue.push(0);

        int index = 0; // start at one for alignment
        while (!queue.empty())
        {
            int nodeIndex = queue.front();
            queue.pop();

            assert(index < mNodes.size());
            assert(nodeIndex < mNodes.size());

            BVHNode *node = &mNodes[nodeIndex];

            // update bounds
            glm::vec3 nodeMin = glm::vec3(std::numeric_limits<float>::max(), std::numeric_limits<float>::max(),
                                          std::numeric_limits<float>::max());
            glm::vec3 nodeMax = glm::vec3(std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest(),
                                          std::numeric_limits<float>::lowest());

            int startIndex = node->mStartIndex;
            int endIndex = node->mStartIndex + node->mIndexCount;

            for (int i = startIndex; i < endIndex; i++)
            {
                const AABB *aabb = &boundingAABBs[mPerm[i]];
                nodeMin = glm::min(nodeMin, aabb->getMin());
                nodeMax = glm::max(nodeMax, aabb->getMax());
            }

            node->mMin = nodeMin;
            node->mMax = nodeMax;

            // Find split (by splitting along longest axis)
            glm::vec3 size = node->mMax - node->mMin;
            int splitPlane = 0;
            if (size.y > size.x && size.y > size.z)
            {
                splitPlane = 1;
            }
            else if (size.z > size.x && size.z > size.y)
            {
                splitPlane = 2;
            }
            float splitPosition = node->mMin[splitPlane] + 0.5f * size[splitPlane];

            // Split bounding AABB's to the left or right of split position
            int i = node->mStartIndex;
            int j = i + node->mIndexCount - 1;
            while (i <= j)
            {
                if (boundingAABBs[mPerm[i]].mCentre[splitPlane] < splitPosition)
                {
                    i++;
                }
                else
                {
                    int temp = mPerm[i];
                    mPerm[i] = mPerm[j];
                    mPerm[j] = temp;
                    j--;
                }
            }

            int leftChildIndexCount = i - node->mStartIndex;
            int rightChildIndexCount = node->mIndexCount - (i - node->mStartIndex);

            if (leftChildIndexCount != 0 && rightChildIndexCount != 0)
            {
                int leftChildIndex = ++index;
                int rightChildIndex = ++index;

                assert(index < mNodes.size());

                mNodes[leftChildIndex].mStartIndex = node->mStartIndex;
                mNodes[leftChildIndex].mIndexCount = leftChildIndexCount;

                mNodes[rightChildIndex].mStartIndex = i;
                mNodes[rightChildIndex].mIndexCount = rightChildIndexCount;

                node->mLeft = leftChildIndex;
                node->mIndexCount = 0;

                queue.push(leftChildIndex);
                queue.push(rightChildIndex);
            }
        }
    }
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
    std::vector<OcclusionQuery> mOcclusionQueries[2];
    int mOcclusionQuery;

    // RenderQueue
    std::vector<std::pair<uint64_t, int>> mRenderQueueScratch;
    std::vector<std::pair<uint64_t, int>> mRenderQueue;

    // Draw call data
    std::vector<glm::mat4> mModels;
    std::vector<Id> mTransformIds;
    std::vector<Sphere> mBoundingSpheres;

    std::vector<DrawCallCommand> mDrawCallCommands;

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
    void update();

    const BVH &getBVH() const;

  private:
    void registerRenderAssets();
    void cacheRenderData();
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