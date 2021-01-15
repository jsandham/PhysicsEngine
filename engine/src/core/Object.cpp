#include "../../include/core/Object.h"

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

Guid Object::getId() const
{
	return mId;
}