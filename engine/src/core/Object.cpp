#include "../../include/core/Object.h"
#include "../../include/core/GLM.h"
#include "../../include/core/Log.h"

using namespace PhysicsEngine;

Object::Object(World *world, const Id& id) : mWorld(world), mGuid(Guid::INVALID), mId(id), mHide(HideFlag::None)
{
}

Object::Object(World *world, const Guid& guid, const Id& id) : mWorld(world), mGuid(guid), mId(id), mHide(HideFlag::None)
{
}

Object::~Object()
{
}

void Object::serialize(YAML::Node &out) const
{
    out["type"] = getType();
    out["hide"] = mHide;
    out["id"] = mGuid;
}

void Object::deserialize(const YAML::Node &in)
{
    mHide = YAML::getValue<HideFlag>(in, "hide");
    mGuid = YAML::getValue<Guid>(in, "id");
}

Guid Object::getGuid() const
{
    return mGuid;
}

Id Object::getId() const
{
    return mId;
}