#include "../../include/components/BoxCollider.h"

#include "../../include/core/Intersect.h"
#include "../../include/core/Serialization.h"

using namespace PhysicsEngine;

BoxCollider::BoxCollider(World* world) : Collider(world)
{
}

BoxCollider::BoxCollider(World* world, Guid id) : Collider(world, id)
{
}

BoxCollider::~BoxCollider()
{
}

void BoxCollider::serialize(YAML::Node &out) const
{
    Collider::serialize(out);

    out["AABB"] = mAABB;
}

void BoxCollider::deserialize(const YAML::Node &in)
{
    Collider::deserialize(in);

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