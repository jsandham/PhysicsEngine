#ifndef MATERIAL_UTIL_H__
#define MATERIAL_UTIL_H__

#include "Material.h"
#include "World.h"

namespace PhysicsEngine
{
class MaterialUtil
{
  public:
    static void copyMaterialTo(World *srcWorld, Material *srcMat, World *destWorld, Material *destMat);
};
} // namespace PhysicsEngine

#endif
