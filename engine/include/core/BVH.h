#ifndef BVH_H__
#define BVH_H__

#include <vector>
#include <queue>

#include "../core/glm.h"
#include "../core/AABB.h"
#include "../core/Ray.h"
#include "../core/Intersect.h"

namespace PhysicsEngine
{
struct BVHNode
{
    glm::vec3 mMin;
    glm::vec3 mMax;
    // If mIndexCount == 0 (i.e. not a leaf node) then mLeftOrStartIndex is the left child index
    // If mIndexCount != 0 (i.e. a leaf node) then mLeftOrStartIndex is the start index
    int mLeftOrStartIndex;
    int mIndexCount;

    inline bool isLeaf() const
    {
        return mIndexCount > 0;
    }
};

struct BVH
{
    BVHNode *mNodes;
    int *mPerm;
    size_t mSize;

    inline int getNodeCount() const
    {
        return (2 * (int)mSize - 1);
    }

    void allocateBVH(size_t size)
    {
        mSize = size;

        if (mSize > 0)
        {
            mNodes = (BVHNode*)malloc(sizeof(BVHNode) * (2 * mSize - 1));
            mPerm = (int*)malloc(sizeof(int) * mSize);
        }
    }

    void freeBVH()
    {
        if (mSize > 0)
        {
            mSize = 0;
            free(mNodes);
            free(mPerm);
        }
    }

    void buildBVH(const AABB* boundingAABBs, size_t size)
    {
        assert(mSize == size);

        if (mSize == 0)
        {
            return;
        }

        for (size_t i = 0; i < mSize; i++)
        {
            mPerm[i] = (int)i;
        }

        mNodes[0].mLeftOrStartIndex = 0;
        mNodes[0].mIndexCount = static_cast<int>(mSize);

        glm::vec3 MAX_VEC3 = glm::vec3(std::numeric_limits<float>::max(), std::numeric_limits<float>::max(),
                                                std::numeric_limits<float>::max());
        glm::vec3 MIN_VEC3 = glm::vec3(std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest(),
                      std::numeric_limits<float>::lowest());

        std::queue<int> queue;
        queue.push(0);

        int index = 0; // start at one for alignment
        while (!queue.empty())
        {
            int nodeIndex = queue.front();
            queue.pop();

            assert(index < getNodeCount());
            assert(nodeIndex < getNodeCount());

            BVHNode *node = &mNodes[nodeIndex];

            // update bounds
            glm::vec3 nodeMin = MAX_VEC3;
            glm::vec3 nodeMax = MIN_VEC3;

            int startIndex = node->mLeftOrStartIndex;
            int endIndex = node->mLeftOrStartIndex + node->mIndexCount;

            for (int i = startIndex; i < endIndex; i++)
            {
                const AABB *aabb = &boundingAABBs[mPerm[i]];
                nodeMin = glm::min(nodeMin, aabb->getMin());
                nodeMax = glm::max(nodeMax, aabb->getMax());
            }

            node->mMin = nodeMin;
            node->mMax = nodeMax;

            // Find split (by splitting along longest axis)
            glm::vec3 aabbSize = node->mMax - node->mMin;
            int splitPlane = 0;
            if (aabbSize.y > aabbSize.x && aabbSize.y > aabbSize.z)
            {
                splitPlane = 1;
            }
            else if (aabbSize.z > aabbSize.x && aabbSize.z > aabbSize.y)
            {
                splitPlane = 2;
            }
            float splitPosition = node->mMin[splitPlane] + 0.5f * aabbSize[splitPlane];

            // Split bounding AABB's to the left or right of split position
            int i = node->mLeftOrStartIndex;
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

            int leftChildIndexCount = i - node->mLeftOrStartIndex;
            int rightChildIndexCount = node->mIndexCount - leftChildIndexCount;

            if (leftChildIndexCount != 0 && rightChildIndexCount != 0)
            {
                int leftChildIndex = ++index;
                int rightChildIndex = ++index;

                assert(index < getNodeCount());

                mNodes[leftChildIndex].mLeftOrStartIndex = node->mLeftOrStartIndex;
                mNodes[leftChildIndex].mIndexCount = leftChildIndexCount;

                mNodes[rightChildIndex].mLeftOrStartIndex = i;
                mNodes[rightChildIndex].mIndexCount = rightChildIndexCount;

                node->mLeftOrStartIndex = leftChildIndex;
                node->mIndexCount = 0;

                queue.push(leftChildIndex);
                queue.push(rightChildIndex);
            }
        }
    }
};
}

#endif