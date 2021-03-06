#include "../../include/core/Object.h"
#include "../../include/core/Serialization.h"

using namespace PhysicsEngine;

Object::Object()
{
    mId = Guid::INVALID;
}

Object::Object(Guid id) : mId(id)
{
}

Object::~Object()
{
}

void Object::serialize(std::ostream &out) const
{
    PhysicsEngine::write<Guid>(out, mId);
}

void Object::deserialize(std::istream &in)
{
    PhysicsEngine::read<Guid>(in, mId);
}

void Object::serialize(YAML::Node& out) const
{
    out["type"] = getType();
    out["id"] = mId;
}

void Object::deserialize(const YAML::Node& in)
{
    mId = in["id"].as<Guid>();
}

Guid Object::getId() const
{
    return mId;
}