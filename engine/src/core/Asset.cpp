#include "../../include/core/Asset.h"
#include "../../include/core/World.h"
#include "../../include/core/Serialize.h"

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

void Asset::serialize(std::ostream& out) const
{
    Object::serialize(out);
    PhysicsEngine::write<std::string>(out, mAssetName);
}

void Asset::deserialize(std::istream& in)
{
    Object::deserialize(in);
    PhysicsEngine::read<std::string>(in, mAssetName);
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