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
    PhysicsEngine::write<HideFlag>(out, mHide);
    PhysicsEngine::write<Guid>(out, mId);
}

void Object::deserialize(std::istream &in)
{
    PhysicsEngine::read<HideFlag>(in, mHide);
    PhysicsEngine::read<Guid>(in, mId);
}

void Object::serialize(YAML::Node &out) const
{
    out["type"] = getType();
    out["hide"] = mHide;
    out["id"] = mId;
}

void Object::deserialize(const YAML::Node &in)
{
    mHide = YAML::getValue<HideFlag>(in, "hide");
    mId = YAML::getValue<Guid>(in, "id");
}

Guid Object::getId() const
{
    return mId;
}