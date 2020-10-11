#include "../../include/core/Frustum.h"

using namespace PhysicsEngine;

Frustum::Frustum()
{
    mFov = 45.0f;
    mAspectRatio = 1.0f;
    mNearPlane = 0.1f;
    mFarPlane = 250.0f;
}

Frustum::~Frustum()
{
}

void Frustum::computePlanes(glm::vec3 position, glm::vec3 front, glm::vec3 up, glm::vec3 right)
{
    front = glm::normalize(front);
    up = glm::normalize(up);
    right = glm::normalize(right);

    float tan = (float)glm::tan(glm::radians(0.5f * mFov));
    float nearPlaneHeight = mNearPlane * tan;
    float nearPlaneWidth = mAspectRatio * nearPlaneHeight;

    // far and near plane intersection along front line
    glm::vec3 fc = position + mFarPlane * front;
    glm::vec3 nc = position + mNearPlane * front;

    mPlanes[NEAR].mNormal = front;
    mPlanes[NEAR].mX0 = nc;

    mPlanes[FAR].mNormal = -front;
    mPlanes[FAR].mX0 = fc;

    glm::vec3 temp;

    // construct normals so that they point inwards
    temp = (nc + nearPlaneHeight * up) - position;
    temp = glm::normalize(temp);
    mPlanes[TOP].mNormal = -glm::cross(temp, right);
    mPlanes[TOP].mX0 = nc + nearPlaneHeight * up;

    temp = (nc - nearPlaneHeight * up) - position;
    temp = glm::normalize(temp);
    mPlanes[BOTTOM].mNormal = glm::cross(temp, right);
    mPlanes[BOTTOM].mX0 = nc - nearPlaneHeight * up;

    temp = (nc - nearPlaneWidth * right) - position;
    temp = glm::normalize(temp);
    mPlanes[LEFT].mNormal = -glm::cross(temp, up);
    mPlanes[LEFT].mX0 = nc - nearPlaneWidth * right;

    temp = (nc + nearPlaneWidth * right) - position;
    temp = glm::normalize(temp);
    mPlanes[RIGHT].mNormal = glm::cross(temp, up);
    mPlanes[RIGHT].mX0 = nc + nearPlaneWidth * right;

    // near corners
    mNtl = nc + nearPlaneHeight * up - nearPlaneWidth * right;
    mNtr = nc + nearPlaneHeight * up + nearPlaneWidth * right;
    mNbl = nc - nearPlaneHeight * up - nearPlaneWidth * right;
    mNbr = nc - nearPlaneHeight * up + nearPlaneWidth * right;

    // far corners
    float farPlaneHeight = mFarPlane * tan;
    float farPlaneWidth = mAspectRatio * farPlaneHeight;

    mFtl = fc + farPlaneHeight * up - farPlaneWidth * right;
    mFtr = fc + farPlaneHeight * up + farPlaneWidth * right;
    mFbl = fc - farPlaneHeight * up - farPlaneWidth * right;
    mFbr = fc - farPlaneHeight * up + farPlaneWidth * right;
}

bool Frustum::containsPoint(glm::vec3 point) const
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