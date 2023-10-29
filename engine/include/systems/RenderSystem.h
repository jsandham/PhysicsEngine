#ifndef RENDERSYSTEM_H__
#define RENDERSYSTEM_H__

#include <vector>
#include <limits>

#include "../core/SerializationEnums.h"
#include "../core/Guid.h"
#include "../core/Id.h"
#include "../core/Input.h"
#include "../core/Time.h"
#include "../core/AABB.h"

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
    int mRight;
    int mStartIndex;
    int mIndexCount;
};

struct BVH
{
    std::vector<int> mPerm;
    std::vector<BVHNode> mNodes;

    void buildBVH(const std::vector<AABB> &boundingAABBs)
    {
        if (boundingAABBs.size() == 0)
        {
            return;
        }

        mPerm.resize(boundingAABBs.size(), 0);
        for (size_t i = 0; i < mPerm.size(); i++)
        {
            mPerm[i] = (int)i;
        }

        mNodes.resize(2 * boundingAABBs.size() - 1);

        mNodes[0].mLeft = 0;
        mNodes[0].mRight = 0;
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

            // update bounds
            glm::vec3 nodeMin = glm::vec3(std::numeric_limits<float>::max(), std::numeric_limits<float>::max(),
                                          std::numeric_limits<float>::max());
            glm::vec3 nodeMax = glm::vec3(std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest(),
                                          std::numeric_limits<float>::lowest());

            for (size_t i = mNodes[nodeIndex].mStartIndex; i < (mNodes[nodeIndex].mStartIndex + mNodes[nodeIndex].mIndexCount); i++)
            {
                glm::vec3 min = boundingAABBs[mPerm[i]].getMin();
                glm::vec3 max = boundingAABBs[mPerm[i]].getMax();


                nodeMin = glm::min(nodeMin, boundingAABBs[mPerm[i]].getMin());
                nodeMax = glm::max(nodeMax, boundingAABBs[mPerm[i]].getMax());
            }

            mNodes[nodeIndex].mMin = nodeMin;
            mNodes[nodeIndex].mMax = nodeMax;

            // Find split (by splitting along longest axis)
            glm::vec3 extents = mNodes[nodeIndex].mMax - mNodes[nodeIndex].mMin;
            int splitPlane = 0;
            if (extents.y > extents.x && extents.y > extents.z)
            {
                splitPlane = 1;
            }
            else if (extents.z > extents.x && extents.z > extents.y)
            {
                splitPlane = 2;
            }
            float splitPosition = mNodes[nodeIndex].mMin[splitPlane] + 0.5f * extents[splitPlane];

            // Split bounding AABB's to the left or right of split position
            int i = mNodes[nodeIndex].mStartIndex;
            int j = i + mNodes[nodeIndex].mIndexCount - 1;
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

            int leftChildIndexCount = i - mNodes[nodeIndex].mStartIndex;
            int rightChildIndexCount = mNodes[nodeIndex].mIndexCount - (i - mNodes[nodeIndex].mStartIndex);

            if (leftChildIndexCount != 0 && rightChildIndexCount != 0)
            {
                int leftChildIndex = ++index;
                int rightChildIndex = ++index;

                assert(index < mNodes.size());

                mNodes[leftChildIndex].mStartIndex = mNodes[nodeIndex].mStartIndex;
                mNodes[leftChildIndex].mIndexCount = leftChildIndexCount;

                mNodes[rightChildIndex].mStartIndex = i;
                mNodes[rightChildIndex].mIndexCount = rightChildIndexCount;

                mNodes[nodeIndex].mLeft = leftChildIndex;
                mNodes[nodeIndex].mRight = rightChildIndex;
                mNodes[nodeIndex].mIndexCount = 0;

                queue.push(leftChildIndex);
                queue.push(rightChildIndex);
            }
        }
    }
};



//struct DrawCallData
//{
//    std::vector<RenderObject> mDrawCalls;
//    std::vector<glm::mat4> mModels;
//    std::vector<Id> mTransformIds;
//    std::vector<Sphere> mBoundingSpheres;
//
//    // std::vector<RenderObject> mInstancedDrawCalls;
//    // std::vector<glm::mat4> mInstancedModels;
//    // std::vector<Id> mInstancedTransformIds;
//    // std::vector<Sphere> mInstancedBoundingSpheres;
//};

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

    // Scratch arrays
    std::vector<RenderObject> mDrawCallScratch;
    std::vector<int> mDrawCallMeshRendererIndices;

    // Frustum culling
    BVH mBVH;
    std::vector<bool> mFrustumVisible;

    // Draw call data
    std::vector<RenderObject> mDrawCalls;
    std::vector<glm::mat4> mModels;
    std::vector<Id> mTransformIds;
    std::vector<Sphere> mBoundingSpheres;

    //std::vector<RenderObject> mInstancedDrawCalls;
    //std::vector<glm::mat4> mInstancedModels;
    //std::vector<Id> mInstancedTransformIds;
    //std::vector<Sphere> mInstancedBoundingSpheres;

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

    const BVH &getBVH() const;

  private:
    void registerRenderAssets(World *world);
    void cacheRenderData(World *world);
    void frustumCulling(World *world, Camera *camera);
    void buildRenderObjectsList(World *world, Camera* camera);
    void buildRenderQueue();
    void sortRenderQueue();
    Sphere computeWorldSpaceBoundingSphere(const glm::mat4 &model, const Sphere &sphere);
    Sphere computeWorldSpaceBoundingSphere(const glm::vec3 &translation, const glm::vec3 &scale, const Sphere &sphere);
    Sphere computeWorldSpaceBoundingSphere(const glm::mat4 &model, const glm::vec3 &scale, const Sphere &sphere);
};

} // namespace PhysicsEngine

#endif