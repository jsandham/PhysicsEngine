#ifndef INTERNAL_MATERIALS_H__
#define INTERNAL_MATERIALS_H__

#include <string>

#include "World.h"

namespace PhysicsEngine
{
class InternalMaterials
{
  public:
    static const std::string simpleLitMaterialName;
    static const std::string colorMaterialName;

    enum class Material
    {
        SimpleLit,
        Color
    };

    template<Material M>
    static Guid loadMaterial(World* world, const Guid& shaderId)
    {
        return Guid::INVALID;
    }

    template<>
    static Guid loadMaterial<Material::SimpleLit>(World* world, const Guid& shaderId)
    {
        return loadInternalMaterial(world, simpleLitMaterialName, shaderId);
    }

    template<>
    static Guid loadMaterial<Material::Color>(World* world, const Guid& shaderId)
    {
        return loadInternalMaterial(world, colorMaterialName, shaderId);
    }

  private:
    static Guid loadInternalMaterial(World *world, const std::string &name, const Guid shaderId);
};
} // namespace PhysicsEngine

#endif