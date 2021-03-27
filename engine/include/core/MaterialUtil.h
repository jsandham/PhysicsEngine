#ifndef MATERIAL_UTIL_H__
#define MATERIAL_UTIL_H__

namespace PhysicsEngine
{
class World;
class Material;

class MaterialUtil
{
  public:
    static void copyMaterialTo(World *srcWorld, Material *srcMat, World *destWorld, Material *destMat);
};
} // namespace PhysicsEngine

#endif
