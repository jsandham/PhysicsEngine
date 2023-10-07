#include <math.h>

#include "../../include/components/SphereCollider.h"

#include "../../include/core/SerializationYaml.h"
#include "../../include/core/World.h"
#include "../../include/core/Intersect.h"

using namespace PhysicsEngine;

void SphereColliderData::serialize(YAML::Node &out) const
{
    out["sphere"] = mSphere;
}

void SphereColliderData::deserialize(const YAML::Node &in)
{
    mSphere = YAML::getValue<Sphere>(in, "sphere");
}

SphereCollider::SphereCollider(World *world, const Id &id) : mWorld(world), mGuid(Guid::INVALID), mId(id), mHide(HideFlag::None)
{
    mEntityGuid = Guid::INVALID;
    mEnabled = true;
}

SphereCollider::SphereCollider(World *world, const Guid &guid, const Id &id) : mWorld(world), mGuid(guid), mId(id), mHide(HideFlag::None)
{
    mEntityGuid = Guid::INVALID;
    mEnabled = true;
}

SphereCollider::~SphereCollider()
{
}

void SphereCollider::serialize(YAML::Node &out) const
{
    out["type"] = getType();
    out["hide"] = mHide;
    out["id"] = mGuid;

    out["entityId"] = mEntityGuid;

    out["enabled"] = mEnabled;
}

void SphereCollider::deserialize(const YAML::Node &in)
{
    mHide = YAML::getValue<HideFlag>(in, "hide");
    mGuid = YAML::getValue<Guid>(in, "id");

    mEntityGuid = YAML::getValue<Guid>(in, "entityId");

    mEnabled = YAML::getValue<bool>(in, "enabled");
}

int SphereCollider::getType() const
{
    return PhysicsEngine::SPHERECOLLIDER_TYPE;
}

std::string SphereCollider::getObjectName() const
{
    return PhysicsEngine::SPHERECOLLIDER_NAME;
}

Guid SphereCollider::getEntityGuid() const
{
    return mEntityGuid;
}

Guid SphereCollider::getGuid() const
{
    return mGuid;
}

Id SphereCollider::getId() const
{
    return mId;
}

bool SphereCollider::intersect(AABB aabb) const
{
    return Intersect::intersect(mSphere, aabb);
}

std::vector<float> SphereCollider::getLines() const
{
    std::vector<float> lines;

    float pi = 3.14159265f;

    for (int i = 0; i < 36; i++)
    {
        float theta1 = i * 10.0f;
        float theta2 = (i + 1) * 10.0f;

        lines.push_back(mSphere.mCentre.x + mSphere.mRadius * cos((pi / 180.0f) * theta1));
        lines.push_back(mSphere.mCentre.y + mSphere.mRadius * sin((pi / 180.0f) * theta1));
        lines.push_back(mSphere.mCentre.z);
        lines.push_back(mSphere.mCentre.x + mSphere.mRadius * cos((pi / 180.0f) * theta2));
        lines.push_back(mSphere.mCentre.y + mSphere.mRadius * sin((pi / 180.0f) * theta2));
        lines.push_back(mSphere.mCentre.z);
    }

    for (int i = 0; i < 36; i++)
    {
        float theta1 = i * 10.0f;
        float theta2 = (i + 1) * 10.0f;

        lines.push_back(mSphere.mCentre.x);
        lines.push_back(mSphere.mCentre.y + mSphere.mRadius * sin((pi / 180.0f) * theta1));
        lines.push_back(mSphere.mCentre.z + mSphere.mRadius * cos((pi / 180.0f) * theta1));
        lines.push_back(mSphere.mCentre.x);
        lines.push_back(mSphere.mCentre.y + mSphere.mRadius * sin((pi / 180.0f) * theta2));
        lines.push_back(mSphere.mCentre.z + mSphere.mRadius * cos((pi / 180.0f) * theta2));
    }

    for (int i = 0; i < 36; i++)
    {
        float theta1 = i * 10.0f;
        float theta2 = (i + 1) * 10.0f;

        lines.push_back(mSphere.mCentre.x + mSphere.mRadius * cos((pi / 180.0f) * theta1));
        lines.push_back(mSphere.mCentre.y);
        lines.push_back(mSphere.mCentre.z + mSphere.mRadius * sin((pi / 180.0f) * theta1));
        lines.push_back(mSphere.mCentre.x + mSphere.mRadius * cos((pi / 180.0f) * theta2));
        lines.push_back(mSphere.mCentre.y);
        lines.push_back(mSphere.mCentre.z + mSphere.mRadius * sin((pi / 180.0f) * theta2));
    }

    return lines;
}