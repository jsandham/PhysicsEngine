#include "../../include/core/Id.h"

using namespace PhysicsEngine;

const Id Id::INVALID = Id(-1);

Id::Id() : mId(-1)
{
  
}

Id::Id(int id) : mId(id)
{
}

Id::~Id()
{

}

Id &Id::operator=(const Id &id)
{
    if (this != &id)
    {
        mId = id.mId;
    }

    return *this;
}

bool Id::operator==(const Id &id) const
{
    return mId == id.mId;
}

bool Id::operator!=(const Id &id) const
{
    return mId != id.mId;
}

bool Id::operator<(const Id &id) const
{
    return mId < id.mId;
}

bool Id::isValid() const
{
    return *this != Id::INVALID;
}

bool Id::isInvalid() const
{
    return *this == Id::INVALID;
}

Id Id::newId()
{
    static int id = 0;
    return id++;
}