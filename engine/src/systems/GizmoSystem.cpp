#include "../../include/systems/GizmoSystem.h"

#include "../../include/core/Serialization.h"
#include "../../include/core/World.h"

using namespace PhysicsEngine;

GizmoSystem::GizmoSystem(World* world) : System(world)
{
}

GizmoSystem::GizmoSystem(World* world, const Guid& id) : System(world, id)
{
}

GizmoSystem::~GizmoSystem()
{
}

void GizmoSystem::serialize(YAML::Node &out) const
{
    System::serialize(out);
}

void GizmoSystem::deserialize(const YAML::Node &in)
{
    System::deserialize(in);
}

int GizmoSystem::getType() const
{
    return PhysicsEngine::GIZMOSYSTEM_TYPE;
}

std::string GizmoSystem::getObjectName() const
{
    return PhysicsEngine::GIZMOSYSTEM_NAME;
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

        if (camera->mEnabled)
        {
            if (camera->mGizmos == CameraGizmos::Gizmos_On)
            {
                mGizmoRenderer.update(camera);
            }

            mGizmoRenderer.drawGrid(camera);
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

void GizmoSystem::addToDrawList(const Sphere& sphere, const Color& color)
{
    mGizmoRenderer.addToDrawList(sphere, color);
}

void GizmoSystem::addToDrawList(const AABB &aabb, const Color &color, bool wireframe)
{
    mGizmoRenderer.addToDrawList(aabb, color, wireframe);
}

void GizmoSystem::addToDrawList(const Frustum &frustum, const Color &color, bool wireframe)
{
    mGizmoRenderer.addToDrawList(frustum, color, wireframe);
}

void GizmoSystem::addToDrawList(const Plane &plane, const glm::vec3 &extents, const Color &color, bool wireframe)
{
    mGizmoRenderer.addToDrawList(plane, extents, color, wireframe);
}

void GizmoSystem::clearDrawList()
{
    mGizmoRenderer.clearDrawList();
}