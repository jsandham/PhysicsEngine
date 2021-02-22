#include "../../include/systems/GizmoSystem.h"

#include "../../include/core/Serialization.h"
#include "../../include/core/World.h"

using namespace PhysicsEngine;

GizmoSystem::GizmoSystem() : System()
{
}

GizmoSystem::GizmoSystem(Guid id) : System(id)
{
}

GizmoSystem::~GizmoSystem()
{
}

void GizmoSystem::serialize(std::ostream &out) const
{
    System::serialize(out);
}

void GizmoSystem::deserialize(std::istream &in)
{
    System::deserialize(in);
}

void GizmoSystem::init(World *world)
{
    mWorld = world;

    mGizmoRenderer.init(mWorld);
}

void GizmoSystem::update(const Input &input, const Time &time)
{
    for (size_t i = 0; i < mWorld->getNumberOfComponents<Camera>(); i++)
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