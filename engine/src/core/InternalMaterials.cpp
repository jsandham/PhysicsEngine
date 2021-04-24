#include "../../include/core/InternalMaterials.h"
#include "../../include/core/World.h"

using namespace PhysicsEngine;

const std::string InternalMaterials::simpleLitMaterialName = "SimpleLit";
const std::string InternalMaterials::colorMaterialName = "Color";

Guid InternalMaterials::loadInternalMaterial(World *world, const std::string &name,
                                             const Guid shaderId)
{
    PhysicsEngine::Material *material = world->createAsset<PhysicsEngine::Material>();
    if (material != nullptr)
    {
        material->setShaderId(shaderId);
        material->setName(name);
        return material->getId();
    }
    
    return Guid::INVALID;
}