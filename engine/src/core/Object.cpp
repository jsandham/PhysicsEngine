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

void Object::serialize(std::ostream& out) const
{
	PhysicsEngine::write<Guid>(out, mId);
}

void Object::deserialize(std::istream& in)
{
	PhysicsEngine::read<Guid>(in, mId);
}

Guid Object::getId() const
{
	return mId;
}