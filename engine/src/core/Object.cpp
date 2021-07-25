#include "../../include/core/Object.h"
#include "../../include/core/Serialization.h"

using namespace PhysicsEngine;

Object::Object(World* world)
{
    mWorld = world;
    mHide = HideFlag::None;
    mId = Guid::INVALID;
}

Object::Object(World* world, Guid id) : mId(id), mHide(HideFlag::None)
{
    mWorld = world;
}

Object::~Object()
{
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