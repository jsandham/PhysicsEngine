#include "../../include/core/Asset.h"
#include "../../include/core/Serialization.h"
#include "../../include/core/World.h"

using namespace PhysicsEngine;

Asset::Asset() : Object()
{
    mName = "Unnamed Asset";
}

Asset::Asset(Guid id) : Object(id)
{
    mName = "Unnamed Asset";
}

Asset::~Asset()
{
}

void Asset::serialize(std::ostream &out) const
{
    Object::serialize(out);
    PhysicsEngine::write<std::string>(out, mName);
}

void Asset::deserialize(std::istream &in)
{
    Object::deserialize(in);
    PhysicsEngine::read<std::string>(in, mName);
}

void Asset::serialize(YAML::Node& out) const
{
    Object::serialize(out);
    out["name"] = mName;
}

void Asset::deserialize(const YAML::Node& in)
{
    Object::deserialize(in);
    mName = in["name"].as<std::string>();
}

std::string Asset::getName() const
{
    return mName;
}

void Asset::setName(const std::string &name)
{
    mName = name;
}

bool Asset::isInternal(int type)
{
    return type >= PhysicsEngine::MIN_INTERNAL_ASSET && type <= PhysicsEngine::MAX_INTERNAL_ASSET;
}