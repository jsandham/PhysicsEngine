#include "../../include/systems/GizmoSystem.h"

#include "../../include/core/SerializationYaml.h"
#include "../../include/core/World.h"

using namespace PhysicsEngine;

GizmoSystem::GizmoSystem(World *world, const Id &id) : mWorld(world), mGuid(Guid::INVALID), mId(id), mHide(HideFlag::None)
{
    mEnabled = true;
}

GizmoSystem::GizmoSystem(World *world, const Guid &guid, const Id &id) : mWorld(world), mGuid(guid), mId(id), mHide(HideFlag::None)
{
    mEnabled = true;
}

GizmoSystem::~GizmoSystem()
{
}

void GizmoSystem::serialize(YAML::Node &out) const
{
    out["type"] = getType();
    out["hide"] = mHide;
    out["id"] = mGuid;
}

void GizmoSystem::deserialize(const YAML::Node &in)
{
    mHide = YAML::getValue<HideFlag>(in, "hide");
    mGuid = YAML::getValue<Guid>(in, "id");
}

int GizmoSystem::getType() const
{
    return PhysicsEngine::GIZMOSYSTEM_TYPE;
}

std::string GizmoSystem::getObjectName() const
{
    return PhysicsEngine::GIZMOSYSTEM_NAME;
}

Guid GizmoSystem::getGuid() const
{
    return mGuid;
}

Id GizmoSystem::getId() const
{
    return mId;
}

void GizmoSystem::init(World *world)
{
    mWorld = world;

    mGizmoRenderer.init(mWorld);
}

void GizmoSystem::update(const Input &input, const Time &time)
{
    for (size_t i = 0; i < mWorld->getActiveScene()->getNumberOfComponents<Camera>(); i++)
    {
        Camera *camera = mWorld->getActiveScene()->getComponentByIndex<Camera>(i);

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

void GizmoSystem::addToDrawList(const Sphere &sphere, const Color &color)
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