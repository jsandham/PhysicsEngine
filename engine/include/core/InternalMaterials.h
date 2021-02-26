#ifndef INTERNAL_MATERIALS_H__
#define INTERNAL_MATERIALS_H__

#include <string>

#include "Guid.h"
#include "Material.h"

namespace PhysicsEngine
{
class InternalMaterials
{
  public:
    static const Guid simpleLitMaterialId;
    static const Guid colorMaterialId;

    static const std::string simpleLitMaterialName;
    static const std::string colorMaterialName;

    static Guid loadSimpleLitMaterial(World *world, const Guid shaderId);
    static Guid loadColorMaterial(World *world, const Guid shaderId);

  private:
    static Guid loadInternalMaterial(World *world, const Guid materialId, const std::string &name, const Guid shaderId);
};
} // namespace PhysicsEngine

#endif