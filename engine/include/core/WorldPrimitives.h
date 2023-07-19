#ifndef WORLD_PRIMITIVES_H__
#define WORLD_PRIMITIVES_H__

#include "Guid.h"

namespace PhysicsEngine
{
enum class PrimitiveType
{
    Plane,
    Disc,
    Cube,
    Sphere,
    Cylinder,
    Cone
};

class World;

struct WorldPrimitives
{
    Guid mPlaneMeshGuid;
    Guid mDiscMeshGuid;
    Guid mCubeMeshGuid;
    Guid mSphereMeshGuid;
    Guid mCylinderMeshGuid;
    Guid mConeMeshGuid;

    Guid mStandardShaderGuid;
    Guid mStandardMaterialGuid;

    void createPrimitiveMeshes(World *world, int nx, int nz);
};
} // namespace PhysicsEngine

#endif