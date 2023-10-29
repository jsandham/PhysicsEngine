#include "../../include/systems/GizmoSystem.h"

#include "../../include/core/SerializationYaml.h"
#include "../../include/core/World.h"
#include "../../include/core/Input.h"
#include "../../include/core/Intersect.h"

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
                //drawSphereIntersectionTest(); // debugging intersection tests

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


void GizmoSystem::drawSphereIntersectionTest()
{
    //static Sphere sphere(glm::vec3(0.0f, 0.0f, 0.0f), 1.0f);
    static AABB aabb(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(1.0f, 1.0f, 1.0f));

    if (PhysicsEngine::getKey(getInput(), KeyCode::F))
    {
        //sphere.mCentre.x -= 0.1f;
        aabb.mCentre.x -= 0.1f;
    }
    if (PhysicsEngine::getKey(getInput(), KeyCode::H))
    {
        //sphere.mCentre.x += 0.1f;
        aabb.mCentre.x += 0.1f;
    }

    if (PhysicsEngine::getKey(getInput(), KeyCode::T))
    {
        //sphere.mCentre.z += 0.1f;
        aabb.mCentre.z += 0.1f;
    }
    if (PhysicsEngine::getKey(getInput(), KeyCode::G))
    {
        //sphere.mCentre.z -= 0.1f;
        aabb.mCentre.z -= 0.1f;
    }

    if (PhysicsEngine::getKey(getInput(), KeyCode::Y))
    {
        //sphere.mRadius += 0.1f;
        aabb.mSize += glm::vec3(0.1f, 0.1f, 0.1f);
    }
    if (PhysicsEngine::getKey(getInput(), KeyCode::U))
    {
        //sphere.mRadius = glm::max(sphere.mRadius - 0.1f, 0.1f);
        aabb.mSize -= glm::vec3(0.1f, 0.1f, 0.1f);
    }





    //mGizmoRenderer.addToDrawList(sphere, Color(0.0f, 0.0f, 1.0f, 0.3f));
    mGizmoRenderer.addToDrawList(aabb, Color(0.0f, 0.0f, 1.0f, 0.3f));





    /*if (PhysicsEngine::getKey(getInput(), KeyCode::Key1))
    {
        mGizmoRenderer.addToDrawList(sphere, Color(0.0f, 0.0f, 1.0f, 0.3f));   
    }
    else if ()
    {
        mGizmoRenderer.addToDrawList(sphere, Color(0.0f, 0.0f, 1.0f, 0.3f));
    }*/














    std::vector<Sphere> spheres;
    spheres.resize(10 * 10);

    for (int z = -5; z < 5; z++)
    {
        for (int x = -5; x < 5; x++)
        {
            spheres[10 * (z + 5) + (x + 5)].mCentre = glm::vec3(2 * x, 0.0f, 2 * z);
            spheres[10 * (z + 5) + (x + 5)].mRadius = 0.5f;
        }
    }

    for (size_t i = 0; i < spheres.size(); i++)
    {
        /*if (Intersect::intersect(sphere, spheres[i]))*/
        if (Intersect::intersect(spheres[i], aabb))
        {
            mGizmoRenderer.addToDrawList(spheres[i], Color(1.0f, 0.0f, 0.0f, 0.3f));
        }
        else
        {
            mGizmoRenderer.addToDrawList(spheres[i], Color(0.0f, 0.0f, 1.0f, 0.3f));
        }
    }

    std::vector<AABB> aabbs;
    aabbs.resize(10 * 10);

    for (int z = -5; z < 5; z++)
    {
        for (int x = -5; x < 5; x++)
        {
            aabbs[10 * (z + 5) + (x + 5)].mCentre = glm::vec3(2 * x, 0.0f, 2 * z + 20);
            aabbs[10 * (z + 5) + (x + 5)].mSize = glm::vec3(1.0f, 1.0f, 1.0f);
        }
    }

    for (size_t i = 0; i < aabbs.size(); i++)
    {
        /*if (Intersect::intersect(sphere, aabbs[i]))*/
        if (Intersect::intersect(aabb, aabbs[i]))
        {
            mGizmoRenderer.addToDrawList(aabbs[i], Color(1.0f, 0.0f, 0.0f, 0.3f));
        }
        else
        {
            mGizmoRenderer.addToDrawList(aabbs[i], Color(0.0f, 0.0f, 1.0f, 0.3f));
        }
    }


    std::vector<Frustum> frustums;
    frustums.resize(10 * 10);

    for (int z = -5; z < 5; z++)
    {
        for (int x = -5; x < 5; x++)
        {
            frustums[10 * (z + 5) + (x + 5)].mFov = 45.0f;
            frustums[10 * (z + 5) + (x + 5)].mAspectRatio = 1.0f;
            frustums[10 * (z + 5) + (x + 5)].mNearPlane = 0.1f;
            frustums[10 * (z + 5) + (x + 5)].mFarPlane = 1.0f;
            frustums[10 * (z + 5) + (x + 5)].computePlanes(glm::vec3(2 * x + 20, 0.0f, 2 * z),
                                                           glm::vec3(0.0f, 0.0f, 1.0f), 
                                                             glm::vec3(0.0f, 1.0f, 0.0f),
                                                           glm::vec3(1.0f, 0.0f, 0.0f));
        }
    }

    for (size_t i = 0; i < frustums.size(); i++)
    {
        /*if (Intersect::intersectExact(sphere, frustums[i]))*/
        if (Intersect::intersect2(aabb, frustums[i]))
        {
            mGizmoRenderer.addToDrawList(frustums[i], Color(1.0f, 0.0f, 0.0f, 0.3f));
        }
        else
        {
            mGizmoRenderer.addToDrawList(frustums[i], Color(0.0f, 0.0f, 1.0f, 0.3f));
        }
    }
}

void GizmoSystem::drawAABBIntersectionTest()
{

}