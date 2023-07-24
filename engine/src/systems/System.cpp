// #include "../../include/systems/System.h"
// #include "../../include/core/GLM.h"

// using namespace PhysicsEngine;

// System::System(World *world, const Id &id) : Object(world, id)
// {
//     mOrder = 0;
//     mEnabled = true;
// }

// System::System(World *world, const Guid &guid, const Id &id) : Object(world, guid, id)
// {
//     mOrder = 0;
//     mEnabled = true;
// }

// System::~System()
// {
// }

// void System::serialize(YAML::Node &out) const
// {
//     Object::serialize(out);

//     out["order"] = mOrder;
//     out["enabled"] = mEnabled;
// }

// void System::deserialize(const YAML::Node &in)
// {
//     Object::deserialize(in);

//     mOrder = YAML::getValue<size_t>(in, "order");
//     mEnabled = YAML::getValue<bool>(in, "enabled");
// }

// size_t System::getOrder() const
// {
//     return mOrder;
// }

// bool System::isInternal(int type)
// {
//     return type >= PhysicsEngine::MIN_INTERNAL_SYSTEM && type <= PhysicsEngine::MAX_INTERNAL_SYSTEM;
// }