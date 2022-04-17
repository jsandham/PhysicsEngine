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
        Id mPlaneMeshId;
        Id mDiscMeshId;
        Id mCubeMeshId;
        Id mSphereMeshId;
        Id mCylinderMeshId;
        Id mConeMeshId;

        Id mStandardShaderId;
        Id mStandardMaterialId;

        void createPrimitiveMeshes(World* world, int nx, int nz);
	};
}

#endif