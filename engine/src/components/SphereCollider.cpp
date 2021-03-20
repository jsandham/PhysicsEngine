#include <math.h>

#include "../../include/components/SphereCollider.h"

#include "../../include/core/Intersect.h"
#include "../../include/core/Serialization.h"

using namespace PhysicsEngine;

SphereCollider::SphereCollider() : Collider()
{
}

SphereCollider::SphereCollider(Guid id) : Collider(id)
{
}

SphereCollider::~SphereCollider()
{
}

void SphereCollider::serialize(std::ostream &out) const
{
    Collider::serialize(out);

    PhysicsEngine::write<Sphere>(out, mSphere);
}

void SphereCollider::deserialize(std::istream &in)
{
    Collider::deserialize(in);

    PhysicsEngine::read<Sphere>(in, mSphere);
}

void SphereCollider::serialize(YAML::Node& out) const
{
    Collider::serialize(out);

    out["sphere"] = mSphere;
}

void SphereCollider::deserialize(const YAML::Node& in)
{
    Collider::deserialize(in);

    mSphere = YAML::getValue<Sphere>(in, "sphere");
}

int SphereCollider::getType() const
{
    return PhysicsEngine::SPHERECOLLIDER_TYPE;
}

std::string SphereCollider::getObjectName() const
{
    return PhysicsEngine::SPHERECOLLIDER_NAME;
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