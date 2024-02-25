#ifndef FRUSTUM_H__
#define FRUSTUM_H__

#undef NEAR
#undef FAR
#undef near
#undef far

#include "glm.h"
#include "Plane.h"

namespace PhysicsEngine
{
class Frustum
{
  public:
    // normals in frustum plane are assumed to point inward
    Plane mPlanes[6];

    // near corner points of frustum
    glm::vec3 mNtl;
    glm::vec3 mNtr;
    glm::vec3 mNbl;
    glm::vec3 mNbr;

    // far corner points of frustum
    glm::vec3 mFtl;
    glm::vec3 mFtr;
    glm::vec3 mFbl;
    glm::vec3 mFbr;

    float mFov;         // vertical fov
    float mAspectRatio; // determines horizontal fov
    float mNearPlane;
    float mFarPlane;

    enum
    {
        TOP = 0,
        BOTTOM,
        LEFT,
        RIGHT,
        NEAR,
        FAR
    };

  public:
    Frustum();
    Frustum(float fov, float aspectRatio, float near, float far);

    void computePlanes(const glm::vec3 &position, const glm::vec3 &front, const glm::vec3 &up, const glm::vec3 &right);
    bool containsPoint(const glm::vec3 &point) const;
};
} // namespace PhysicsEngine

#endif