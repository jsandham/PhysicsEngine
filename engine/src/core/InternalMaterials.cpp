#include "../../include/core/InternalMaterials.h"
#include "../../include/core/Log.h"
#include "../../include/core/World.h"

using namespace PhysicsEngine;

const Guid InternalMaterials::simpleLitMaterialId("b62cfdf8-7296-4a5b-9713-e83108a7348d");
const Guid InternalMaterials::colorMaterialId("1a641451-0490-403c-a957-c0a738cdc01b");

const std::string InternalMaterials::simpleLitMaterialName = "SimpleLit";
const std::string InternalMaterials::colorMaterialName = "Color";

Guid InternalMaterials::loadInternalMaterial(World *world, const Guid materialId, const std::string &name, const Guid shaderId)
{
    // Create temp material to compute serialized data vector
    Material temp;
    temp.load(shaderId);
    temp.setName(name);

    std::vector<char> data = temp.serialize(materialId);

    Material *material = world->createAsset<Material>(data);
    if (material != NULL)
    {
        return material->getId();
    }
    else
    {
        Log::error("Could not load internal material\n");
        return Guid::INVALID;
    }
}

Guid InternalMaterials::loadSimpleLitMaterial(World *world, const Guid shaderId)
{
    return loadInternalMaterial(world, InternalMaterials::simpleLitMaterialId, InternalMaterials::simpleLitMaterialName, shaderId);
}

Guid InternalMaterials::loadColorMaterial(World *world, const Guid shaderId)
{
    return loadInternalMaterial(world, InternalMaterials::colorMaterialId, InternalMaterials::colorMaterialName, shaderId);
}
