#include "../../include/components/BoxCollider.h"

#include "../../include/core/Intersect.h"
#include "../../include/core/Serialize.h"

using namespace PhysicsEngine;

BoxCollider::BoxCollider() : Collider()
{
}

BoxCollider::BoxCollider(Guid id) : Collider(id)
{
}

BoxCollider::~BoxCollider()
{
}

std::vector<char> BoxCollider::serialize() const
{
    return serialize(mId, mEntityId);
}

std::vector<char> BoxCollider::serialize(const Guid &componentId, const Guid &entityId) const
{
    BoxColliderHeader header;
    header.mComponentId = componentId;
    header.mEntityId = entityId;
    header.mAABB = mAABB;

    std::vector<char> data(sizeof(BoxColliderHeader));

    memcpy(&data[0], &header, sizeof(BoxColliderHeader));

    return data;
}

void BoxCollider::deserialize(const std::vector<char> &data)
{
    const BoxColliderHeader *header = reinterpret_cast<const BoxColliderHeader *>(&data[0]);

    mId = header->mComponentId;
    mEntityId = header->mEntityId;
    mAABB = header->mAABB;
}

void BoxCollider::serialize(std::ostream& out) const
{
    Collider::serialize(out);
    PhysicsEngine::write<AABB>(out, mAABB);
}

void BoxCollider::deserialize(std::istream& in)
{
    Collider::deserialize(in);
    PhysicsEngine::read<AABB>(in, mAABB);
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