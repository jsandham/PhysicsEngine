#ifndef RTGEOMETRY_H__
#define RTGEOMETRY_H__

#include <vector>

#include "BVH.h"

namespace PhysicsEngine
{
class Mesh;

struct RTGeometryHit
{
    TLASHit mHit;
};

class RTGeometry
{
  private:
    TLAS mTLAS;
    std::vector<BLAS *> mBLAS;

    void *mdTLASNodes;
    void *mdPerm;
    void *mdBLASNodes;
    void *mdBLASPerm;
    void *mdBLASTriangles;
    void *mdModels;
    void *mdInverseModels;

  public:
    RTGeometry();

    void createRTGeometry();
    void destroyRTGeometry();
    void buildRTGeometryAccelStruct(const std::vector<Mesh *> &meshes, const std::vector<glm::mat4> &models);
    RTGeometryHit intersect(const Ray &ray, int intersectCount) const;
    glm::vec3 getTriangleWorldSpaceUnitNormal(const RTGeometryHit &hit) const;


    // void turnDebugOn();
    // void turnDebugOff();
};
} // namespace PhysicsEngine

#endif
