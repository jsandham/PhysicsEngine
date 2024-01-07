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

    float hit_sphere(const glm::vec3 &center, float radius, const Ray &ray) const
    {
        glm::vec3 oc = ray.mOrigin - center;
        auto a = glm::dot(ray.mDirection, ray.mDirection);
        auto half_b = glm::dot(oc, ray.mDirection);
        auto c = glm::dot(oc, oc) - radius * radius;
        auto discriminant = half_b * half_b - a * c;

        if (discriminant < 0)
        {
            return -1.0;
        }
        else
        {
            return (-half_b - sqrt(discriminant)) / a;
        }
    }

    void intersectBVH(const Ray &ray, const std::vector<Sphere> &spheres, const int nodeIndex, float &closest_t, int &closest_index) const
    {
        const BVHNode *node = &mNodes[nodeIndex];

        AABB aabb;
        aabb.mCentre = 0.5f * (node->mMax + node->mMin);
        aabb.mSize = node->mMax - node->mMin;

        if (!Intersect::intersect(ray, aabb))
        {
            return;
        }

        if (node->isLeaf())
        {
            int startIndex = node->mStartIndex;
            int endIndex = node->mStartIndex + node->mIndexCount;
            for (int i = startIndex; i < endIndex; i++)
            {
                float t = hit_sphere(spheres[mPerm[i]].mCentre, spheres[mPerm[i]].mRadius, ray);
                if (t > 0.001f && t < closest_t)
                {
                    closest_t = t;
                    closest_index = (int)mPerm[i];
                }
            }
        }
        else
        {
            intersectBVH(ray, spheres, node->mLeft, closest_t, closest_index);
            intersectBVH(ray, spheres, node->mLeft + 1, closest_t, closest_index);
        }
    }
};
}

#endif