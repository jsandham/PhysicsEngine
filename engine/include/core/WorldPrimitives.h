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
        Guid mPlaneMeshId;
        Guid mDiscMeshId;
        Guid mCubeMeshId;
        Guid mSphereMeshId;
        Guid mCylinderMeshId;
        Guid mConeMeshId;

        Guid mStandardShaderId;
        Guid mStandardMaterialId;

        void createPrimitiveMeshes(World* world, int nx, int nz);
	};
}

#endif