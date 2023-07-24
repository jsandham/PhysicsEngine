// #include "../../include/components/Component.h"
// #include "../../include/core/World.h"

// using namespace PhysicsEngine;

// Component::Component(World *world, const Id &id) : Object(world, id)
// {
//     mEntityGuid = Guid::INVALID;
// }

// Component::Component(World *world, const Guid &guid, const Id &id) : Object(world, guid, id)
// {
//     mEntityGuid = Guid::INVALID;
// }

// Component::~Component()
// {
// }

// void Component::serialize(YAML::Node &out) const
// {
//     Object::serialize(out);

//     out["entityId"] = mEntityGuid;
// }

// void Component::deserialize(const YAML::Node &in)
// {
//     Object::deserialize(in);

//     mEntityGuid = YAML::getValue<Guid>(in, "entityId");
// }

// Entity *Component::getEntity() const
// {
//     return mWorld->getActiveScene()->getEntityByGuid(mEntityGuid);
// }

// Guid Component::getEntityGuid() const
// {
//     return mEntityGuid;
// }

// bool Component::isInternal(int type)
// {
//     return type >= PhysicsEngine::MIN_INTERNAL_COMPONENT && type <= PhysicsEngine::MAX_INTERNAL_COMPONENT;
// }