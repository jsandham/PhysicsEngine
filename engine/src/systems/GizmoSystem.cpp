#include "../../include/systems/GizmoSystem.h"

#include "../../include/core/World.h"

using namespace PhysicsEngine;

GizmoSystem::GizmoSystem()
{
}

GizmoSystem::GizmoSystem(const std::vector<char> &data)
{
}

GizmoSystem::~GizmoSystem()
{
}

std::vector<char> GizmoSystem::serialize() const
{
    return serialize(mSystemId);
}

std::vector<char> GizmoSystem::serialize(const Guid &systemId) const
{
    GizmoSystemHeader header;
    header.mSystemId = systemId;
    header.mUpdateOrder = static_cast<int32_t>(mOrder);

    std::vector<char> data(sizeof(GizmoSystemHeader));

    memcpy(&data[0], &header, sizeof(GizmoSystemHeader));

    return data;
}

void GizmoSystem::deserialize(const std::vector<char> &data)
{
    const GizmoSystemHeader *header = reinterpret_cast<const GizmoSystemHeader *>(&data[0]);

    mSystemId = header->mSystemId;
    mOrder = static_cast<int>(header->mUpdateOrder);
}

void GizmoSystem::init(World *world)
{
    mWorld = world;

    mGizmoRenderer.init(mWorld);
}

void GizmoSystem::update(const Input &input, const Time &time)
{
    for (int i = 0; i < mWorld->getNumberOfComponents<Camera>(); i++)
    {
        Camera *camera = mWorld->getComponentByIndex<Camera>(i);

        if (camera->mGizmos == CameraGizmos::Gizmos_On)
        {
            mGizmoRenderer.update(camera);
        }
    }
}

void GizmoSystem::addToDrawList(const Line &line, const Color &color)
{
    mGizmoRenderer.addToDrawList(line, color);
}

void GizmoSystem::addToDrawList(const Ray &ray, float t, const Color &color)
{
    mGizmoRenderer.addToDrawList(ray, t, color);
}

void GizmoSystem::addToDrawList(const AABB &aabb, const Color &color)
{
    mGizmoRenderer.addToDrawList(aabb, color);
}

void GizmoSystem::addToDrawList(const Sphere &sphere, const Color &color)
{
    mGizmoRenderer.addToDrawList(sphere, color);
}

void GizmoSystem::addToDrawList(const Frustum &frustum, const Color &color)
{
    mGizmoRenderer.addToDrawList(frustum, color);
}

void GizmoSystem::addToDrawList(const Plane &plane, const glm::vec3 &extents, const Color &color)
{
    mGizmoRenderer.addToDrawList(plane, extents, color);
}

void GizmoSystem::clearDrawList()
{
    mGizmoRenderer.clearDrawList();
}