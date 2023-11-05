#include "../../include/core/Frustum.h"

using namespace PhysicsEngine;

Frustum::Frustum() : mFov(45.0f), mAspectRatio(1.0f), mNearPlane(0.1f), mFarPlane(250.0f)
{
}

Frustum::Frustum(float fov, float aspectRatio, float near, float far)
    : mFov(fov), mAspectRatio(aspectRatio), mNearPlane(near), mFarPlane(far)
{
}

void Frustum::computePlanes(const glm::vec3 &position, const glm::vec3 &front, const glm::vec3 &up,
                            const glm::vec3 &right)
{
    assert(mNearPlane > 0.0f);
    assert(mNearPlane < mFarPlane);
    
    glm::vec3 nfront = glm::normalize(front);
    glm::vec3 nup = glm::normalize(up);
    glm::vec3 nright = glm::normalize(right);

    float tan = (float)glm::tan(glm::radians(0.5f * mFov));
    float nearPlaneHeight = mNearPlane * tan;
    float nearPlaneWidth = mAspectRatio * nearPlaneHeight;

    // far and near plane intersection along front line
    glm::vec3 fc = position + mFarPlane * nfront;
    glm::vec3 nc = position + mNearPlane * nfront;

    mPlanes[NEAR].mNormal = nfront;
    mPlanes[NEAR].mX0 = nc;

    mPlanes[FAR].mNormal = -nfront;
    mPlanes[FAR].mX0 = fc;

    glm::vec3 temp;

    // construct normals so that they point inwards
    temp = (nc + nearPlaneHeight * nup) - position;
    temp = glm::normalize(temp);
    mPlanes[TOP].mNormal = -glm::cross(temp, nright);
    mPlanes[TOP].mX0 = nc + nearPlaneHeight * nup;

    temp = (nc - nearPlaneHeight * nup) - position;
    temp = glm::normalize(temp);
    mPlanes[BOTTOM].mNormal = glm::cross(temp, nright);
    mPlanes[BOTTOM].mX0 = nc - nearPlaneHeight * nup;

    temp = (nc - nearPlaneWidth * nright) - position;
    temp = glm::normalize(temp);
    mPlanes[LEFT].mNormal = -glm::cross(temp, nup);
    mPlanes[LEFT].mX0 = nc - nearPlaneWidth * nright;

    temp = (nc + nearPlaneWidth * nright) - position;
    temp = glm::normalize(temp);
    mPlanes[RIGHT].mNormal = glm::cross(temp, nup);
    mPlanes[RIGHT].mX0 = nc + nearPlaneWidth * nright;

    // near corners
    mNtl = nc + nearPlaneHeight * nup - nearPlaneWidth * nright;
    mNtr = nc + nearPlaneHeight * nup + nearPlaneWidth * nright;
    mNbl = nc - nearPlaneHeight * nup - nearPlaneWidth * nright;
    mNbr = nc - nearPlaneHeight * nup + nearPlaneWidth * nright;

    // far corners
    float farPlaneHeight = mFarPlane * tan;
    float farPlaneWidth = mAspectRatio * farPlaneHeight;

    mFtl = fc + farPlaneHeight * nup - farPlaneWidth * nright;
    mFtr = fc + farPlaneHeight * nup + farPlaneWidth * nright;
    mFbl = fc - farPlaneHeight * nup - farPlaneWidth * nright;
    mFbr = fc - farPlaneHeight * nup + farPlaneWidth * nright;
}

bool Frustum::containsPoint(const glm::vec3 &point) const
{
    // point lies outside frustum
    if (mPlanes[0].signedDistance(point) < 0)
    {
        return false;
    }
    if (mPlanes[1].signedDistance(point) < 0)
    {
        return false;
    }
    if (mPlanes[2].signedDistance(point) < 0)
    {
        return false;
    }
    if (mPlanes[3].signedDistance(point) < 0)
    {
        return false;
    }
    if (mPlanes[4].signedDistance(point) < 0)
    {
        return false;
    }
    if (mPlanes[5].signedDistance(point) < 0)
    {
        return false;
    }

    // point lies on or inside frustum
    return 1;
}