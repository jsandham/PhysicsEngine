#include "../../include/components/BoxCollider.h"

#include "../../include/core/SerializationYaml.h"
#include "../../include/core/Intersect.h"
#include "../../include/core/World.h"

using namespace PhysicsEngine;

BoxCollider::BoxCollider(World *world, const Id &id) : mWorld(world), mGuid(Guid::INVALID), mId(id), mHide(HideFlag::None)
{
    mEntityGuid = Guid::INVALID;
    mEnabled = true;
}

BoxCollider::BoxCollider(World *world, const Guid &guid, const Id &id) : mWorld(world), mGuid(guid), mId(id), mHide(HideFlag::None)
{
    mEntityGuid = Guid::INVALID;
    mEnabled = true;
}

BoxCollider::~BoxCollider()
{
}

void BoxCollider::serialize(YAML::Node &out) const
{
    out["type"] = getType();
    out["hide"] = mHide;
    out["id"] = mGuid;

    out["entityId"] = mEntityGuid;

    out["enabled"] = mEnabled;

    out["AABB"] = mAABB;
}

void BoxCollider::deserialize(const YAML::Node &in)
{
    mHide = YAML::getValue<HideFlag>(in, "hide");
    mGuid = YAML::getValue<Guid>(in, "id");

    mEntityGuid = YAML::getValue<Guid>(in, "entityId");

    mEnabled = YAML::getValue<bool>(in, "enabled");

    mAABB = YAML::getValue<AABB>(in, "AABB");
}

int BoxCollider::getType() const
{
    return PhysicsEngine::BOXCOLLIDER_TYPE;
}

std::string BoxCollider::getObjectName() const
{
    return PhysicsEngine::BOXCOLLIDER_NAME;
}

Guid BoxCollider::getEntityGuid() const
{
    return mEntityGuid;
}

Guid BoxCollider::getGuid() const
{
    return mGuid;
}

Id BoxCollider::getId() const
{
    return mId;
}

bool BoxCollider::intersect(AABB aabb) const
{
    return Intersect::intersect(mAABB, aabb);
}

std::vector<float> BoxCollider::getLines()
    const // might not want to store lines in class so if I end up doing that, instead move this to a utility method??
{
    std::vector<float> lines;
    glm::vec3 centre = mAABB.mCentre;
    glm::vec3 extents = mAABB.getExtents();

    float xf[] = {-1.0f, 1.0f, 1.0f, 1.0f, 1.0f, -1.0f, -1.0f, -1.0f};
    float yf[] = {1.0f, 1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f};
    float zf[] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};

    for (int i = 0; i < 8; i++)
    {
        lines.push_back(centre.x + xf[i] * extents.x);
        lines.push_back(centre.y + yf[i] * extents.y);
        lines.push_back(centre.z + zf[i] * extents.z);
    }

    for (int i = 0; i < 8; i++)
    {
        lines.push_back(centre.x + xf[i] * extents.x);
        lines.push_back(centre.y + yf[i] * extents.y);
        lines.push_back(centre.z - zf[i] * extents.z);
    }

    float xg[] = {-1.0f, -1.0f, 1.0f, 1.0f, 1.0f, 1.0f, -1.0f, -1.0f};
    float yg[] = {1.0f, 1.0f, 1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f};
    float zg[] = {-1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f};

    for (int i = 0; i < 8; i++)
    {
        lines.push_back(centre.x + xg[i] * extents.x);
        lines.push_back(centre.y + yg[i] * extents.y);
        lines.push_back(centre.z + zg[i] * extents.z);
    }

    // lines.push_back(centre.x - extents.x);
    // lines.push_back(centre.y - extents.y);
    // lines.push_back(centre.z + extents.z);

    // lines.push_back(centre.x + extents.x);
    // lines.push_back(centre.y - extents.y);
    // lines.push_back(centre.z + extents.z);

    // lines.push_back(centre.x + extents.x);
    // lines.push_back(centre.y - extents.y);
    // lines.push_back(centre.z + extents.z);

    // lines.push_back(centre.x + extents.x);
    // lines.push_back(centre.y + extents.y);
    // lines.push_back(centre.z + extents.z);

    // lines.push_back(centre.x + extents.x);
    // lines.push_back(centre.y + extents.y);
    // lines.push_back(centre.z + extents.z);

    // lines.push_back(centre.x - extents.x);
    // lines.push_back(centre.y + extents.y);
    // lines.push_back(centre.z + extents.z);

    // lines.push_back(centre.x - extents.x);
    // lines.push_back(centre.y + extents.y);
    // lines.push_back(centre.z + extents.z);

    // lines.push_back(centre.x - extents.x);
    // lines.push_back(centre.y - extents.y);
    // lines.push_back(centre.z + extents.z);

    return lines;
}