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

struct BVHLeaf
{
    int mStartIndex;
    int mIndexCount;
};

struct BVHHit
{
    BVHLeaf mLeafs[32];
    int mLeafCount;
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

        assert(mNodes != nullptr);
        assert(mPerm != nullptr);

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

    BVHHit intersect(const Ray &ray) const
    {
        BVHHit hit;
        hit.mLeafCount = 0;

        int top = 0;
        int stack[32];

        stack[0] = 0;
        top++;

        while (top > 0)
        {
            int nodeIndex = stack[top - 1];
            top--;

            const BVHNode *node = &mNodes[nodeIndex];

            if (Intersect::intersect(ray, node->mMin, node->mMax))
            {
                if (!node->isLeaf())
                {
                    stack[top++] = node->mLeftOrStartIndex;
                    stack[top++] = node->mLeftOrStartIndex + 1;
                }
                else
                {
                    if (hit.mLeafCount < 32)
                    {
                        hit.mLeafs[hit.mLeafCount].mStartIndex = node->mLeftOrStartIndex;
                        hit.mLeafs[hit.mLeafCount].mIndexCount = node->mIndexCount;
                        hit.mLeafCount++;
                    }
                }
            }
        }

        return hit;
    }
};







































inline float intersectTri(const Triangle &triangle, const Ray &ray)
{
    constexpr float epsilon = std::numeric_limits<float>::epsilon();

    glm::vec3 edge1 = triangle.mV1 - triangle.mV0;
    glm::vec3 edge2 = triangle.mV2 - triangle.mV0;
    glm::vec3 ray_cross_e2 = glm::cross(ray.mDirection, edge2);
    float det = glm::dot(edge1, ray_cross_e2);

    if (det > -epsilon && det < epsilon)
        return -1.0f; // This ray is parallel to this triangle.

    float inv_det = 1.0f / det;
    glm::vec3 s = ray.mOrigin - triangle.mV0;
    float u = inv_det * glm::dot(s, ray_cross_e2);

    if (u < 0 || u > 1)
        return -1.0f;

    glm::vec3 s_cross_e1 = glm::cross(s, edge1);
    float v = inv_det * glm::dot(ray.mDirection, s_cross_e1);

    if (v < 0 || u + v > 1)
        return -1.0f;

    // At this stage we can compute t to find out where the intersection point is on the line.
    float t = inv_det * glm::dot(edge2, s_cross_e1);

    if (t > epsilon) // ray intersection
    {
        return t;
    }
    else // This means that there is a line intersection but not a ray intersection.
        return -1.0f;
}

struct BLASHit
{
    int mTriIndex;
    float mT;
};

struct BLAS
{
    BVHNode *mNodes;
    int *mPerm;
    size_t mSize;
    Triangle *mTriangles;
    glm::mat4 mModel;
    glm::mat4 mInverseModel;

    inline int getNodeCount() const
    {
        return (2 * (int)mSize - 1);
    }

    inline Triangle getTriangle(size_t index) const
    {
        return mTriangles[index];
    }

    void allocateBLAS(size_t size)
    {
        mSize = size;

        if (mSize > 0)
        {
            mNodes = (BVHNode *)malloc(sizeof(BVHNode) * (2 * mSize - 1));
            mPerm = (int *)malloc(sizeof(int) * mSize);
            mTriangles = (Triangle *)malloc(sizeof(Triangle) * mSize);
        }
    }

    void freeBLAS()
    {
        if (mSize > 0)
        {
            mSize = 0;
            free(mNodes);
            free(mPerm);
            free(mTriangles);
        }
    }

    void buildBLAS(const std::vector<Triangle> &triangles, const glm::mat4 &model, size_t size)
    {
        assert(mSize == size);

        if (mSize == 0)
        {
            return;
        }

        assert(mNodes != nullptr);
        assert(mPerm != nullptr);
        assert(mTriangles != nullptr);

        mModel = model;
        mInverseModel = glm::inverse(model);

        std::vector<AABB> boundingAABBs(mSize);
        for (size_t i = 0; i < mSize; i++)
        {
            mTriangles[i] = triangles[i];
            boundingAABBs[i] = triangles[i].getAABBBounds();
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

    BLASHit intersectBLAS(const Ray &ray) const
    {
        Ray modelSpaceRay;
        modelSpaceRay.mOrigin = mInverseModel * glm::vec4(ray.mOrigin, 1.0f);
        modelSpaceRay.mDirection = mInverseModel * glm::vec4(ray.mDirection, 0.0f);

        BLASHit hit;
        hit.mTriIndex = -1;
        hit.mT = std::numeric_limits<float>::max();

        int top = 0;
        int stack[32];

        stack[0] = 0;
        top++;

        while (top > 0)
        {
            int nodeIndex = stack[top - 1];
            top--;

            const BVHNode *node = &mNodes[nodeIndex];

            if (Intersect::intersect(modelSpaceRay, node->mMin, node->mMax))
            {
                if (!node->isLeaf())
                {
                    stack[top++] = node->mLeftOrStartIndex;
                    stack[top++] = node->mLeftOrStartIndex + 1;
                }
                else
                {
                    int startIndex = node->mLeftOrStartIndex;
                    int endIndex = node->mLeftOrStartIndex + node->mIndexCount;

                    for (int j = startIndex; j < endIndex; j++)
                    {
                        float t = intersectTri(mTriangles[mPerm[j]], modelSpaceRay);
                        if (t > 0.001f && t < hit.mT)
                        {
                            hit.mT = t;
                            hit.mTriIndex = mPerm[j];
                        }
                    }
                }
            }
        }

        return hit;
    }

    AABB getAABBBounds() const
    {
        glm::vec3 size = mNodes[0].mMax - mNodes[0].mMin;

        glm::vec3 a[8];
        a[0] = mNodes[0].mMin;
        a[1] = mNodes[0].mMin + glm::vec3(size.x, 0.0f, 0.0f);
        a[2] = mNodes[0].mMin + glm::vec3(size.x, size.y, 0.0f);
        a[3] = mNodes[0].mMin + glm::vec3(size.x, 0.0f, size.z);
        a[4] = mNodes[0].mMin + glm::vec3(size.x, size.y, size.z);
        a[5] = mNodes[0].mMin + glm::vec3(0.0f, size.y, 0.0f);
        a[6] = mNodes[0].mMin + glm::vec3(0.0f, size.y, size.z);
        a[7] = mNodes[0].mMin + glm::vec3(0.0f, 0.0f, size.z);

        for (int i = 0; i < 8; i++)
        {
            a[i] = glm::vec3(mModel * glm::vec4(a[i], 1.0f));
        }

        glm::vec3 min = a[0];
        glm::vec3 max = a[0];
        for (int i = 1; i < 8; i++)
        {
            min = glm::min(min, a[i]);
            max = glm::max(max, a[i]);
        }
       
        AABB aabb;
        aabb.mSize = max - min;
        aabb.mCentre = min + 0.5f * aabb.mSize;

        return aabb;
    }
};

struct TLASHit
{
    BLASHit blasHit;
    int blasIndex;
};

struct TLAS
{
    BVHNode *mNodes;
    int *mPerm;
    size_t mSize;
    BLAS *mBlas;

    inline int getNodeCount() const
    {
        return (2 * (int)mSize - 1);
    }

    void allocateTLAS(size_t size)
    {
        mSize = size;

        if (mSize > 0)
        {
            mNodes = (BVHNode *)malloc(sizeof(BVHNode) * (2 * mSize - 1));
            mPerm = (int *)malloc(sizeof(int) * mSize);
        }
    }

    void freeTLAS()
    {
        if (mSize > 0)
        {
            mSize = 0;
            free(mNodes);
            free(mPerm);
        }
    }

    void buildTLAS(BLAS *blas, size_t size)
    {
        assert(mSize == size);

        if (mSize == 0)
        {
            return;
        }

        mBlas = blas;

        assert(mNodes != nullptr);
        assert(mPerm != nullptr);
        assert(mBlas != nullptr);

        std::vector<AABB> boundingAABBs(mSize);
        for (size_t i = 0; i < mSize; i++)
        {
            boundingAABBs[i] = blas[i].getAABBBounds();
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

    TLASHit intersectTLAS(const Ray &ray) const
    {
        TLASHit hit;
        hit.blasHit.mTriIndex = -1;
        hit.blasHit.mT = std::numeric_limits<float>::max();
        hit.blasIndex = -1;

        int top = 0;
        int stack[32];

        stack[0] = 0;
        top++;

        while (top > 0)
        {
            int nodeIndex = stack[top - 1];
            top--;

            const BVHNode *node = &mNodes[nodeIndex];

            if (Intersect::intersect(ray, node->mMin, node->mMax))
            {
                if (!node->isLeaf())
                {
                    stack[top++] = node->mLeftOrStartIndex;
                    stack[top++] = node->mLeftOrStartIndex + 1;
                }
                else
                {
                    int startIndex = node->mLeftOrStartIndex;
                    int endIndex = node->mLeftOrStartIndex + node->mIndexCount;

                    for (int j = startIndex; j < endIndex; j++)
                    {
                        BLASHit h = mBlas[mPerm[j]].intersectBLAS(ray);

                        if (h.mT > 0.001f && h.mT < hit.blasHit.mT)
                        {
                            hit.blasHit.mT = h.mT;
                            hit.blasHit.mTriIndex = h.mTriIndex;
                            hit.blasIndex = mPerm[j];
                        }
                    }
                }
            }
        }

        return hit;
    }
};



















struct TLASNode
{
    glm::vec3 mMin;
    glm::vec3 mMax;
    unsigned int mLeft;
    unsigned int mRight;
    unsigned int mBLAS;
    inline bool isLeaf() const
    {
        return (mLeft == 0 && mRight == 0);
    }
};

struct TLAS2
{
    TLASNode *mNodes;
    size_t mSize;
    BLAS *mBlas;

    void allocateTLAS2(size_t size)
    {
        mSize = size;

        if (mSize > 0)
        {
            mNodes = (TLASNode *)malloc(sizeof(TLASNode) * 2 * mSize);
        }
    }

    void freeTLAS2()
    {
        if (mSize > 0)
        {
            mSize = 0;
            free(mNodes);
        }
    }

    int findSmallestSAMatch(const std::vector<int> &nodeIndices, int N, int nodeIndex)
    {
        float maxSurfaceArea = std::numeric_limits<float>::max();
        int match = -1;

        for (int i = 0; i < N; i++)
        {
            if (nodeIndices[i] != nodeIndex)
            {
                glm::vec3 bmin = glm::min(mNodes[nodeIndices[i]].mMin, mNodes[nodeIndices[nodeIndex]].mMin);
                glm::vec3 bmax = glm::max(mNodes[nodeIndices[i]].mMax, mNodes[nodeIndices[nodeIndex]].mMax);
            
                glm::vec3 bsize = bmax - bmin;
                float surfaceArea = bsize.x * bsize.y + bsize.y * bsize.z + bsize.z * bsize.x;
                if (surfaceArea < maxSurfaceArea)
                {
                    maxSurfaceArea = surfaceArea;
                    match = nodeIndices[i];
                }
            }
        }

        return match;
    }

    void buildTLAS2(BLAS* blas, size_t size)
    {
        assert(mSize == size);

        if (mSize == 0)
        {
            return;
        }

        assert(mNodes != nullptr);
        assert(mBlas != nullptr);

        mBlas = blas;

        std::vector<int> nodeIndices(size);

        int index = 1;
        for (size_t i = 0; i < size; i++)
        {
            nodeIndices[i] = index;

            mNodes[index].mMin = blas[i].getAABBBounds().getMin();
            mNodes[index].mMax = blas[i].getAABBBounds().getMax();
            mNodes[index].mBLAS = (unsigned int)i;
            mNodes[index].mLeft = 0;
            mNodes[index].mRight = 0;
            index++;
        }

        // Find best match to A
        int A = 0;
        int B = findSmallestSAMatch(nodeIndices, (int)nodeIndices.size(), A);

        int count = (int)nodeIndices.size();
        while (count > 1)
        {
            int C = findSmallestSAMatch(nodeIndices, count, B);
        
            if (A == C)
            {
                int A_Idx = nodeIndices[A];
                int B_Idx = nodeIndices[B];

                TLASNode *nodeA = &mNodes[A_Idx];
                TLASNode *nodeB = &mNodes[B_Idx];

                TLASNode *parentAB = &mNodes[index];
                parentAB->mLeft = A_Idx;
                parentAB->mRight = B_Idx;
                parentAB->mMin = glm::min(nodeA->mMin, nodeB->mMin);
                parentAB->mMax = glm::max(nodeA->mMax, nodeB->mMax);

                nodeIndices[A] = index++;
                nodeIndices[B] = nodeIndices[count - 1];
                B = findSmallestSAMatch(nodeIndices, --count, A);
            }
            else
            {
                A = B;
                B = C;
            }
        }

        mNodes[0] = mNodes[nodeIndices[A]];
    }

    TLASHit intersectTLAS2(const Ray &ray) const
    {
        TLASHit hit;
        hit.blasHit.mTriIndex = -1;
        hit.blasHit.mT = std::numeric_limits<float>::max();
        hit.blasIndex = -1;

        int top = 0;
        int stack[32];

        stack[0] = 0;
        top++;

        while (top > 0)
        {
            int nodeIndex = stack[top - 1];
            top--;

            const TLASNode *node = &mNodes[nodeIndex];

            if (Intersect::intersect(ray, node->mMin, node->mMax))
            {
                if (!node->isLeaf())
                {
                    stack[top++] = node->mLeft;
                    stack[top++] = node->mRight;
                }
                else
                {
                    BLASHit h = mBlas[node->mBLAS].intersectBLAS(ray);

                    if (h.mT > 0.001f && h.mT < hit.blasHit.mT)
                    {
                        hit.blasHit.mT = h.mT;
                        hit.blasHit.mTriIndex = h.mTriIndex;
                        hit.blasIndex = node->mBLAS;
                    }
                }
            }
        }

        return hit;
    }
};

}

#endif