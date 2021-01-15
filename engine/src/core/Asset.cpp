#include "../../include/core/Asset.h"
#include "../../include/core/World.h"

using namespace PhysicsEngine;

Asset::Asset() : Object()
{
    mAssetName = "Unnamed Asset";
}

Asset::Asset(Guid id) : Object(id)
{
    mAssetName = "Unnamed Asset";
}

Asset::~Asset()
{
}

std::string Asset::getName() const
{
    return mAssetName;
}

void Asset::setName(const std::string& name)
{
    mAssetName = name;
}

bool Asset::isInternal(int type)
{
    return type >= PhysicsEngine::MIN_INTERNAL_ASSET && type <= PhysicsEngine::MAX_INTERNAL_ASSET;
}