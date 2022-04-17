#include "../../include/core/Object.h"
#include "../../include/core/GLM.h"
#include "../../include/core/World.h"

using namespace PhysicsEngine;

Object::Object(World *world) : mWorld(world), mId(-1), mHide(HideFlag::None)
{
}

Object::Object(World *world, Id id) : mWorld(world), mId(id), mHide(HideFlag::None)
{
}

Object::~Object()
{
}

void Object::serialize(YAML::Node &out) const
{
    out["type"] = getType();
    out["hide"] = mHide;
    out["id"] = mWorld->getGuidOf(mId);
}

void Object::deserialize(const YAML::Node &in)
{
    mHide = YAML::getValue<HideFlag>(in, "hide");
    mId = mWorld->getIdOf(YAML::getValue<Guid>(in, "id"));
}

Guid Object::getGuid() const
{
    return mWorld->getGuidOf(mId);
}

Id Object::getId() const
{
    return mId;
}