#include "../../include/core/Object.h"

using namespace PhysicsEngine;

Object::Object(World *world) : mWorld(world), mId(Guid::INVALID), mHide(HideFlag::None)
{
}

Object::Object(World *world, const Guid& id) : mWorld(world), mId(id), mHide(HideFlag::None)
{
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