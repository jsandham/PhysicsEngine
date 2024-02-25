#include "../../include/core/RTGeometry.h"
#include "../../include/core/Mesh.h"

using namespace PhysicsEngine;

RTGeometry::RTGeometry()
{
    mdTLASNodes = nullptr;
    mdPerm = nullptr;
    mdBLASNodes = nullptr;
    mdBLASPerm = nullptr;
    mdBLASTriangles = nullptr;
    mdModels = nullptr;
    mdInverseModels = nullptr;
}
void RTGeometry::createRTGeometry()
{
}

void RTGeometry::destroyRTGeometry()
{
}

void RTGeometry::buildRTGeometryAccelStruct(const std::vector<Mesh *> &meshes, const std::vector<glm::mat4> &models)
{
    mBLAS.resize(meshes.size());
    for (size_t i = 0; i < meshes.size(); i++)
    {
        mBLAS[i] = meshes[i]->getBLAS();
    }

    mTLAS.buildTLAS(mBLAS, models);
}

RTGeometryHit RTGeometry::intersect(const Ray &ray, int intersectCount) const
{
    RTGeometryHit hit;
    hit.mHit = mTLAS.intersectTLAS(ray, intersectCount);
    return hit;
}

glm::vec3 RTGeometry::getTriangleWorldSpaceUnitNormal(const RTGeometryHit &hit) const
{
    return mBLAS[hit.mHit.blasIndex]->getTriangleWorldSpaceUnitNormal(mTLAS.mModels[hit.mHit.blasIndex],
                                                                      hit.mHit.mTriIndex);
}