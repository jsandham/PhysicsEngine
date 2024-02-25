#ifndef BVH_H__
#define BVH_H__

#include <vector>
#include <queue>
#include <stack>
#include <iostream>

#include "../core/glm.h"
#include "../core/Sphere.h"
#include "../core/AABB.h"
#include "../core/Triangle.h"
#include "../core/Ray.h"

namespace PhysicsEngine
{
inline float intersectSphere(const glm::vec3 &center, float radius, const Ray &ray)
{
    glm::vec3 oc = (ray.mOrigin - center);
    float a = glm::dot(ray.mDirection, ray.mDirection);
    float b = 2.0f * glm::dot(oc, ray.mDirection);
    float c = glm::dot(oc, oc) - radius * radius;
    float discriminant = b * b - 4 * a * c;

    if (discriminant < 0.0f)
    {
        return -1.0f;
    }
    else
    {
        return (-b - glm::sqrt(discriminant)) / (2.0f * a);
    }
}

inline float intersectAABB(const Ray &ray, const glm::vec3 &bmin, const glm::vec3 &bmax)
{
    glm::vec3 invDirection;
    invDirection.x = 1.0f / ray.mDirection.x;
    invDirection.y = 1.0f / ray.mDirection.y;
    invDirection.z = 1.0f / ray.mDirection.z;

    float tx0 = (bmin.x - ray.mOrigin.x) * invDirection.x;
    float tx1 = (bmax.x - ray.mOrigin.x) * invDirection.x;
    float ty0 = (bmin.y - ray.mOrigin.y) * invDirection.y;
    float ty1 = (bmax.y - ray.mOrigin.y) * invDirection.y;
    float tz0 = (bmin.z - ray.mOrigin.z) * invDirection.z;
    float tz1 = (bmax.z - ray.mOrigin.z) * invDirection.z;

    float tmin = glm::max(glm::max(glm::min(tx0, tx1), glm::min(ty0, ty1)), glm::min(tz0, tz1));
    float tmax = glm::min(glm::min(glm::max(tx0, tx1), glm::max(ty0, ty1)), glm::max(tz0, tz1));

    return (tmax >= tmin && tmax >= 0.0f) ? tmin : std::numeric_limits<float>::max();
}

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

struct BVHHit
{
    int mIndex;
    float mT;
};

struct BVH
{
    std::vector<BVHNode> mNodes;
    std::vector<int> mPerm;

    inline int getNodeCount() const
    {
        return (2 * (int)mPerm.size() - 1);
    }

    void buildBVH(const std::vector<AABB> &boundingAABBs)
    {
        if (boundingAABBs.size() == 0)
        {
            return;
        }

        mPerm.resize(boundingAABBs.size());
        for (size_t i = 0; i < boundingAABBs.size(); i++)
        {
            mPerm[i] = (int)i;
        }

        
        mNodes.resize(2 * boundingAABBs.size() - 1);

        mNodes[0].mLeftOrStartIndex = 0;
        mNodes[0].mIndexCount = static_cast<int>(boundingAABBs.size());

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

            assert((startIndex >= 0 && startIndex < mPerm.size()));
            assert((endIndex >= 0 && endIndex <= mPerm.size()));

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
            if (aabbSize.y > aabbSize.x)
                splitPlane = 1;
            if (aabbSize.z > aabbSize[splitPlane])
                splitPlane = 2;
            float splitPosition = node->mMin[splitPlane] + aabbSize[splitPlane] * 0.5f;

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

    BVHHit intersect(const Ray &ray, const std::vector<Sphere> &spheres, int &intersectionCount) const
    {
        BVHHit hit;
        hit.mT = std::numeric_limits<float>::max();
        hit.mIndex = -1;

        float root_t = intersectAABB(ray, mNodes[0].mMin, mNodes[0].mMax);
        intersectionCount++;
        if (root_t == std::numeric_limits<float>::max())
        {
            return hit;
        }

        int top = 0;
        int stack[32];

        stack[0] = 0;
        top++;

        while (top > 0)
        {
            assert(top <= 32);

            int nodeIndex = stack[top - 1];
            top--;

            const BVHNode *node = &mNodes[nodeIndex];

            if (node->isLeaf())
            {
                int startIndex = node->mLeftOrStartIndex;
                int endIndex = node->mLeftOrStartIndex + node->mIndexCount;

                for (int j = startIndex; j < endIndex; j++)
                {
                    float t = intersectSphere(spheres[mPerm[j]].mCentre, spheres[mPerm[j]].mRadius, ray);
                    if (t > 0.001f && t < hit.mT)
                    {
                        hit.mT = t;
                        hit.mIndex = (int)mPerm[j];
                    }
                }
            }
            else
            {
                const BVHNode *left = &mNodes[node->mLeftOrStartIndex];
                const BVHNode *right = &mNodes[node->mLeftOrStartIndex + 1];
                
                float lt = intersectAABB(ray, left->mMin, left->mMax);
                float rt = intersectAABB(ray, right->mMin, right->mMax);
                intersectionCount += 2;

                if (lt <= rt)
                {
                    // Left node is closer than right node. Place right node on stack followed by left node
                    // so that the left node (now top of the stack) will be processed first
                    if (hit.mT > rt)
                    {
                        stack[top++] = node->mLeftOrStartIndex + 1;
                    }
                    if (hit.mT > lt)
                    {
                        stack[top++] = node->mLeftOrStartIndex;
                    }
                }
                else
                {
                    // Right node is closer than left node. Place left node on stack followed by right node
                    // so that the right node (now top of the stack) will be processed first
                    if (hit.mT > lt)
                    {
                        stack[top++] = node->mLeftOrStartIndex;
                    }
                    if (hit.mT > rt)
                    {
                        stack[top++] = node->mLeftOrStartIndex + 1;
                    }
                }
            }
        }

        return hit;
    }
};

struct BLASHit
{
    int mTriIndex;
    float mT;
};

struct BLAS
{
    std::vector<BVHNode> mNodes;
    std::vector<int> mPerm;
    std::vector<Triangle> mTriangles;

    inline int getNodeCount() const
    {
        return (2 * (int)mPerm.size() - 1);
    }

    inline Triangle getTriangle(size_t index) const
    {
        return mTriangles[index];
    }

    inline glm::vec3 getTriangleWorldSpaceNormal(const glm::mat4 &model, size_t index) const
    {
        return glm::vec3(model * glm::vec4(getTriangle(index).getNormal(), 0.0f));
    }

    inline glm::vec3 getTriangleWorldSpaceUnitNormal(const glm::mat4 &model, size_t index) const
    {
        return glm::normalize(getTriangleWorldSpaceNormal(model, index));
    }

    struct Bin
    {
        glm::vec3 mMin;
        glm::vec3 mMax;
        int mTriCount;
    };

    inline float findSAHSplitPlane(const BVHNode *node, int &splitPlane, float &splitPosition)
    {
        constexpr int BIN_COUNT = 8;

        splitPlane = 0;
        float cost = std::numeric_limits<float>::max();

        int startIndex = node->mLeftOrStartIndex;
        int endIndex = node->mLeftOrStartIndex + node->mIndexCount;

        for (int axis = 0; axis < 3; axis++)
        {
            float min = node->mMin[axis];
            float max = node->mMax[axis];

            if (min < max)
            {
                float ds = (max - min) / BIN_COUNT;

                Bin bins[BIN_COUNT];
                for (int i = 0; i < BIN_COUNT; i++)
                {
                    bins[i].mMin = glm::vec3(std::numeric_limits<float>::max(), std::numeric_limits<float>::max(),
                        std::numeric_limits<float>::max());
                    bins[i].mMax = glm::vec3(std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest(),
                        std::numeric_limits<float>::lowest());
                    bins[i].mTriCount = 0;
                }

                assert((startIndex >= 0 && startIndex < mPerm.size()));
                assert((endIndex >= 0 && endIndex <= mPerm.size()));

                for (int i = startIndex; i < endIndex; i++)
                {
                    const Triangle& tri = mTriangles[mPerm[i]];

                    // Find which bin each triangle belongs to
                    int binIdx = glm::min(BIN_COUNT - 1, (int)((tri.getCentroid()[axis] - min) / ds));

                    bins[binIdx].mMin = glm::min(bins[binIdx].mMin, tri.mV0);
                    bins[binIdx].mMin = glm::min(bins[binIdx].mMin, tri.mV1);
                    bins[binIdx].mMin = glm::min(bins[binIdx].mMin, tri.mV2);

                    bins[binIdx].mMax = glm::max(bins[binIdx].mMax, tri.mV0);
                    bins[binIdx].mMax = glm::max(bins[binIdx].mMax, tri.mV1);
                    bins[binIdx].mMax = glm::max(bins[binIdx].mMax, tri.mV2);

                    bins[binIdx].mTriCount++;
                }

                float leftArea[BIN_COUNT - 1];
                float rightArea[BIN_COUNT - 1];
                int leftCount[BIN_COUNT - 1];
                int rightCount[BIN_COUNT - 1];

                glm::vec3 leftMin = glm::vec3(std::numeric_limits<float>::max(), std::numeric_limits<float>::max(),
                    std::numeric_limits<float>::max());
                glm::vec3 leftMax = glm::vec3(std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest(),
                    std::numeric_limits<float>::lowest());
                glm::vec3 rightMin = glm::vec3(std::numeric_limits<float>::max(), std::numeric_limits<float>::max(),
                    std::numeric_limits<float>::max());
                glm::vec3 rightMax = glm::vec3(std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest(),
                    std::numeric_limits<float>::lowest());
                int leftSum = 0;
                int rightSum = 0;
                for (int i = 0; i < (BIN_COUNT - 1); i++)
                {
                    leftMin = glm::min(leftMin, bins[i].mMin);
                    leftMax = glm::max(leftMax, bins[i].mMax);
                    leftSum += bins[i].mTriCount;

                    glm::vec3 leftSize = leftMax - leftMin;
                    leftArea[i] = leftSize.x * leftSize.y + leftSize.y * leftSize.z + leftSize.z * leftSize.x;
                    leftCount[i] = leftSum;

                    rightMin = glm::min(rightMin, bins[BIN_COUNT - 1 - i].mMin);
                    rightMax = glm::max(rightMax, bins[BIN_COUNT - 1 - i].mMax);
                    rightSum += bins[BIN_COUNT - 1 - i].mTriCount;

                    glm::vec3 rightSize = rightMax - rightMin;
                    rightArea[BIN_COUNT - 2 - i] =
                        rightSize.x * rightSize.y + rightSize.y * rightSize.z + rightSize.z * rightSize.x;
                    rightCount[BIN_COUNT - 2 - i] = rightSum;
                }

                for (int i = 0; i < BIN_COUNT - 1; i++)
                {
                    // Surface are heuristic
                    float c = leftCount[i] * leftArea[i] + rightCount[i] * rightArea[i];
                    c = (c > 0.0f) ? c : std::numeric_limits<float>::max();

                    if (c < cost)
                    {
                        cost = c;
                        splitPlane = axis;
                        splitPosition = min + ds * (i + 1);
                    }
                }
            }
        }

        return cost;
    }

    void buildBLAS(const std::vector<Triangle> &triangles)
    {
        if (triangles.size() == 0)
        {
            return;
        }

        mTriangles = triangles;

        mPerm.resize(triangles.size());
        for (size_t i = 0; i < mPerm.size(); i++)
        {
            mPerm[i] = (int)i;
        }

        mNodes.resize(2 * triangles.size() - 1);

        mNodes[0].mLeftOrStartIndex = 0;
        mNodes[0].mIndexCount = static_cast<int>(triangles.size());

        glm::vec3 MAX_VEC3 = glm::vec3(std::numeric_limits<float>::max(), std::numeric_limits<float>::max(),
                                       std::numeric_limits<float>::max());
        glm::vec3 MIN_VEC3 = glm::vec3(std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest(),
                                       std::numeric_limits<float>::lowest());

        int top = 0;
        int stack[32];

        stack[0] = 0;
        top++;

        int index = 0; // start at one for alignment
        while (top > 0)
        {
            assert(top <= 32);

            int nodeIndex = stack[top - 1];
            top--;

            assert(index < getNodeCount());
            assert(nodeIndex < getNodeCount());

            BVHNode *node = &mNodes[nodeIndex];

            // update bounds
            glm::vec3 nodeMin = MAX_VEC3;
            glm::vec3 nodeMax = MIN_VEC3;

            int startIndex = node->mLeftOrStartIndex;
            int endIndex = node->mLeftOrStartIndex + node->mIndexCount;

            if (!(endIndex >= 0 && endIndex <= mPerm.size()) || !(startIndex >= 0 && startIndex < mPerm.size()))
            {
                std::cout << "startIndex: " << startIndex << " endIndex: " << endIndex << std::endl;
            }

            assert((startIndex >= 0 && startIndex < mPerm.size()));
            assert((endIndex >= 0 && endIndex <= mPerm.size()));

            for (int i = startIndex; i < endIndex; i++)
            {
                const Triangle *tri = &triangles[mPerm[i]];
                nodeMin = glm::min(nodeMin, tri->mV0);
                nodeMin = glm::min(nodeMin, tri->mV1);
                nodeMin = glm::min(nodeMin, tri->mV2);

                nodeMax = glm::max(nodeMax, tri->mV0);
                nodeMax = glm::max(nodeMax, tri->mV1);
                nodeMax = glm::max(nodeMax, tri->mV2);
            }

            node->mMin = nodeMin;
            node->mMax = nodeMax;

            //if (node->mIndexCount > 10)
            {
                // Find split (by splitting along longest axis)
                int splitPlane;
                float splitPosition;
                float splitCost = findSAHSplitPlane(node, splitPlane, splitPosition);

                glm::vec3 e = node->mMax - node->mMin; // extent of the node
                float surfaceArea = e.x * e.y + e.y * e.z + e.z * e.x;
                float noSplitCost = node->mIndexCount * surfaceArea;

                if (splitCost < noSplitCost)
                {
                    // Split triangles to the left or right of split position
                    int i = node->mLeftOrStartIndex;
                    int j = i + node->mIndexCount - 1;
                    while (i <= j)
                    {
                        if (triangles[mPerm[i]].getCentroid()[splitPlane] < splitPosition)
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

                        stack[top++] = leftChildIndex;
                        stack[top++] = rightChildIndex;
                    }
                }
            }
        }
    }

    BLASHit intersectBLAS(const Ray &ray, float maxt, int &intersectCount) const
    {
        BLASHit hit;
        hit.mTriIndex = -1;
        hit.mT = maxt;

        float root_t = intersectAABB(ray, mNodes[0].mMin, mNodes[0].mMax);
        intersectCount++;
        if (root_t == std::numeric_limits<float>::max())
        {
            return hit;
        }

        int top = 0;
        int stack[32];

        stack[0] = 0;
        top++;

        while (top > 0)
        {
            assert(top <= 32);

            int nodeIndex = stack[top - 1];
            top--;

            const BVHNode *node = &mNodes[nodeIndex];

            if (node->isLeaf())
            {
                int startIndex = node->mLeftOrStartIndex;
                int endIndex = node->mLeftOrStartIndex + node->mIndexCount;

                for (int j = startIndex; j < endIndex; j++)
                {
                    float t = intersectTri(mTriangles[mPerm[j]], ray);
                    if (t > 0.001f && t < hit.mT)
                    {
                        hit.mT = t;
                        hit.mTriIndex = mPerm[j];
                    }
                }
            }
            else
            {
                const BVHNode *left = &mNodes[node->mLeftOrStartIndex];
                const BVHNode *right = &mNodes[node->mLeftOrStartIndex + 1];
                float lt = intersectAABB(ray, left->mMin, left->mMax);
                float rt = intersectAABB(ray, right->mMin, right->mMax);
                intersectCount += 2;

                if (lt <= rt)
                {
                    // Left node is closer than right node. Place right node on stack followed by left node
                    // so that the left node (now top of the stack) will be processed first
                    if (hit.mT > rt)
                    {
                        stack[top++] = node->mLeftOrStartIndex + 1;
                    }
                    if (hit.mT > lt)
                    {
                        stack[top++] = node->mLeftOrStartIndex;      
                    }
                }
                else
                {
                    // Right node is closer than left node. Place left node on stack followed by right node
                    // so that the right node (now top of the stack) will be processed first
                    if (hit.mT > lt)
                    {
                        stack[top++] = node->mLeftOrStartIndex;
                    }
                    if (hit.mT > rt)
                    {
                        stack[top++] = node->mLeftOrStartIndex + 1;
                    }
                }
            }
        }

        return hit;
    }

    AABB getAABBBounds(const glm::mat4 &model) const
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
            a[i] = glm::vec3(model * glm::vec4(a[i], 1.0f));
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
    int blasIndex;
    int mTriIndex;
    float mT;
};

struct TLAS
{
    std::vector<BVHNode> mNodes;
    std::vector<int> mPerm;
    std::vector<BLAS*> mBLAS;
    std::vector<glm::mat4> mModels;
    std::vector<glm::mat4> mInverseModels;

    inline int getNodeCount() const
    {
        return (2 * (int)mBLAS.size() - 1);
    }

    void buildTLAS(const std::vector<BLAS*> &blas, const std::vector<glm::mat4>& models)
    {
        if (blas.size() == 0)
        {
            return;
        }

        assert(blas.size() == models.size());

        mBLAS = blas;
        mModels = models;

        mInverseModels.resize(mModels.size());
        for (size_t i = 0; i < mInverseModels.size(); i++)
        {
            mInverseModels[i] = glm::inverse(mModels[i]);
        }

        std::vector<AABB> boundingAABBs(mBLAS.size());
        for (size_t i = 0; i < boundingAABBs.size(); i++)
        {
            boundingAABBs[i] = mBLAS[i]->getAABBBounds(mModels[i]);
        }

        mPerm.resize(blas.size());
        for (size_t i = 0; i < mPerm.size(); i++)
        {
            mPerm[i] = (int)i;
        }

        mNodes.resize(2 * blas.size() - 1);

        mNodes[0].mLeftOrStartIndex = 0;
        mNodes[0].mIndexCount = static_cast<int>(blas.size());

        glm::vec3 MAX_VEC3 = glm::vec3(std::numeric_limits<float>::max(), std::numeric_limits<float>::max(),
                                       std::numeric_limits<float>::max());
        glm::vec3 MIN_VEC3 = glm::vec3(std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest(),
                                       std::numeric_limits<float>::lowest());

        int top = 0;
        int stack[32];

        stack[0] = 0;
        top++;

        int index = 0; // start at one for alignment
        while (top > 0)
        {
            assert(top <= 32);

            int nodeIndex = stack[top - 1];
            top--;

            assert(index < getNodeCount());
            assert(nodeIndex < getNodeCount());

            BVHNode *node = &mNodes[nodeIndex];

            // update bounds
            glm::vec3 nodeMin = MAX_VEC3;
            glm::vec3 nodeMax = MIN_VEC3;

            int startIndex = node->mLeftOrStartIndex;
            int endIndex = node->mLeftOrStartIndex + node->mIndexCount;

            assert((startIndex >= 0 && startIndex < mPerm.size()));
            assert((endIndex >= 0 && endIndex <= mPerm.size()));

            for (int i = startIndex; i < endIndex; i++)
            {
                const AABB *aabb = &boundingAABBs[mPerm[i]];
                nodeMin = glm::min(nodeMin, aabb->getMin());
                nodeMax = glm::max(nodeMax, aabb->getMax());
            }

            node->mMin = nodeMin;
            node->mMax = nodeMax;

            if (node->mIndexCount > 2)
            {
                // Find split (by splitting along longest axis)
                glm::vec3 aabbSize = node->mMax - node->mMin;
                int splitPlane = 0;
                if (aabbSize.y > aabbSize.x)
                    splitPlane = 1;
                if (aabbSize.z > aabbSize[splitPlane])
                    splitPlane = 2;
                float splitPosition = node->mMin[splitPlane] + aabbSize[splitPlane] * 0.5f;

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

                    stack[top++] = leftChildIndex;
                    stack[top++] = rightChildIndex;
                }
            }
        }
    }

    TLASHit intersectTLAS(const Ray &ray, int& intersectCount) const
    {
        TLASHit hit;
        hit.blasIndex = -1;
        hit.mTriIndex = -1;
        hit.mT = std::numeric_limits<float>::max();

        float root_t = intersectAABB(ray, mNodes[0].mMin, mNodes[0].mMax);
        intersectCount++;
        if (root_t == std::numeric_limits<float>::max())
        {
            return hit;
        }

        int top = 0;
        int stack[32];

        stack[0] = 0;
        top++;

        while (top > 0)
        {
            assert(top <= 32);

            int nodeIndex = stack[top - 1];
            top--;

            const BVHNode *node = &mNodes[nodeIndex];

            if (node->isLeaf())
            {
                int startIndex = node->mLeftOrStartIndex;
                int endIndex = node->mLeftOrStartIndex + node->mIndexCount;

                for (int j = startIndex; j < endIndex; j++)
                {
                    Ray modelSpaceRay;
                    modelSpaceRay.mOrigin = mInverseModels[mPerm[j]] * glm::vec4(ray.mOrigin, 1.0f);
                    modelSpaceRay.mDirection = mInverseModels[mPerm[j]] * glm::vec4(ray.mDirection, 0.0f);

                    BLASHit h = mBLAS[mPerm[j]]->intersectBLAS(modelSpaceRay, hit.mT, intersectCount);

                    if (h.mT > 0.001f && h.mT < hit.mT)
                    {
                        hit.mT = h.mT;
                        hit.mTriIndex = h.mTriIndex;
                        hit.blasIndex = mPerm[j];
                    }
                }
            }
            else
            {
                const BVHNode *left = &mNodes[node->mLeftOrStartIndex];
                const BVHNode *right = &mNodes[node->mLeftOrStartIndex + 1];
                
                float lt = intersectAABB(ray, left->mMin, left->mMax);
                float rt = intersectAABB(ray, right->mMin, right->mMax);
                intersectCount += 2;

                if (lt <= rt)
                {
                    // Left node is closer than right node. Place right node on stack followed by left node
                    // so that the left node (now top of the stack) will be processed first
                    if (hit.mT > rt)
                    {
                        stack[top++] = node->mLeftOrStartIndex + 1;
                    }
                    if (hit.mT > lt)
                    {
                        stack[top++] = node->mLeftOrStartIndex;
                    }
                }
                else
                {
                    // Right node is closer than left node. Place left node on stack followed by right node
                    // so that the right node (now top of the stack) will be processed first
                    if (hit.mT > lt)
                    {
                        stack[top++] = node->mLeftOrStartIndex;
                    }
                    if (hit.mT > rt)
                    {
                        stack[top++] = node->mLeftOrStartIndex + 1;
                    }
                }
            }
        }

        return hit;
    }
};















//
//
//struct TLASNode
//{
//    glm::vec3 mMin;
//    glm::vec3 mMax;
//    unsigned int mLeft;
//    unsigned int mRight;
//    unsigned int mBLAS;
//    inline bool isLeaf() const
//    {
//        return (mLeft == 0 && mRight == 0);
//    }
//};
//
//struct TLAS2
//{
//    TLASNode *mNodes;
//    size_t mSize;
//    BLAS *mBlas;
//
//    void allocateTLAS2(size_t size)
//    {
//        mSize = size;
//
//        if (mSize > 0)
//        {
//            mNodes = (TLASNode *)malloc(sizeof(TLASNode) * 2 * mSize);
//        }
//    }
//
//    void freeTLAS2()
//    {
//        if (mSize > 0)
//        {
//            mSize = 0;
//            free(mNodes);
//        }
//    }
//
//    int findSmallestSAMatch(const std::vector<int> &nodeIndices, int N, int nodeIndex)
//    {
//        float maxSurfaceArea = std::numeric_limits<float>::max();
//        int match = -1;
//
//        for (int i = 0; i < N; i++)
//        {
//            if (i != nodeIndex)
//            {
//                glm::vec3 bmin = glm::min(mNodes[nodeIndices[i]].mMin, mNodes[nodeIndices[nodeIndex]].mMin);
//                glm::vec3 bmax = glm::max(mNodes[nodeIndices[i]].mMax, mNodes[nodeIndices[nodeIndex]].mMax);
//            
//                glm::vec3 bsize = bmax - bmin;
//                float surfaceArea = bsize.x * bsize.y + bsize.y * bsize.z + bsize.z * bsize.x;
//                if (surfaceArea < maxSurfaceArea)
//                {
//                    maxSurfaceArea = surfaceArea;
//                    match = i;
//                }
//            }
//        }
//
//        return match;
//    }
//
//    void buildTLAS2(BLAS* blas, size_t size)
//    {
//        assert(mSize == size);
//
//        if (mSize == 0)
//        {
//            return;
//        }
//
//        assert(mNodes != nullptr);
//        assert(mBlas != nullptr);
//
//        mBlas = blas;
//
//        std::vector<int> nodeIndices(size);
//
//        int index = 1;
//        for (size_t i = 0; i < size; i++)
//        {
//            nodeIndices[i] = index;
//
//            mNodes[index].mMin = blas[i].getAABBBounds().getMin();
//            mNodes[index].mMax = blas[i].getAABBBounds().getMax();
//            mNodes[index].mBLAS = (unsigned int)i;
//            mNodes[index].mLeft = 0;
//            mNodes[index].mRight = 0;
//            index++;
//        }
//
//        // Find best match to A
//        int A = 0;
//        int B = findSmallestSAMatch(nodeIndices, (int)nodeIndices.size(), A);
//
//        int count = (int)nodeIndices.size();
//        while (count > 1)
//        {
//            int C = findSmallestSAMatch(nodeIndices, count, B);
//        
//            if (A == C)
//            {
//                int A_Idx = nodeIndices[A];
//                int B_Idx = nodeIndices[B];
//
//                TLASNode *nodeA = &mNodes[A_Idx];
//                TLASNode *nodeB = &mNodes[B_Idx];
//
//                TLASNode *parentAB = &mNodes[index];
//                parentAB->mLeft = A_Idx;
//                parentAB->mRight = B_Idx;
//                parentAB->mMin = glm::min(nodeA->mMin, nodeB->mMin);
//                parentAB->mMax = glm::max(nodeA->mMax, nodeB->mMax);
//
//                nodeIndices[A] = index++;
//                nodeIndices[B] = nodeIndices[count - 1];
//                B = findSmallestSAMatch(nodeIndices, --count, A);
//            }
//            else
//            {
//                A = B;
//                B = C;
//            }
//        }
//
//        mNodes[0] = mNodes[nodeIndices[A]];
//    }
//
//    TLASHit intersectTLAS2(const Ray &ray) const
//    {
//        TLASHit hit;
//        hit.blasHit.mTriIndex = -1;
//        hit.blasHit.mT = std::numeric_limits<float>::max();
//        hit.blasIndex = -1;
//
//        int top = 0;
//        int stack[32];
//
//        stack[0] = 0;
//        top++;
//
//        while (top > 0)
//        {
//            int nodeIndex = stack[top - 1];
//            top--;
//
//            const TLASNode *node = &mNodes[nodeIndex];
//
//            if (intersectAABB(ray, node->mMin, node->mMax))
//            {
//                if (!node->isLeaf())
//                {
//                    stack[top++] = node->mLeft;
//                    stack[top++] = node->mRight;
//                }
//                else
//                {
//                    BLASHit h = mBlas[node->mBLAS].intersectBLAS(ray);
//
//                    if (h.mT > 0.001f && h.mT < hit.blasHit.mT)
//                    {
//                        hit.blasHit.mT = h.mT;
//                        hit.blasHit.mTriIndex = h.mTriIndex;
//                        hit.blasIndex = node->mBLAS;
//                    }
//                }
//            }
//        }
//
//        return hit;
//    }
//};

}

#endif