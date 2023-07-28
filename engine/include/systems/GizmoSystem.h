#ifndef GIZMOSYSTEM_H__
#define GIZMOSYSTEM_H__

#include <vector>

#include "../core/SerializationEnums.h"
#include "../core/Guid.h"
#include "../core/Id.h"
#include "../core/AABB.h"
#include "../core/Color.h"
#include "../core/Frustum.h"
#include "../core/Input.h"
#include "../core/Time.h"
#include "../core/Line.h"
#include "../core/Plane.h"
#include "../core/Ray.h"
#include "../core/Sphere.h"

#include "../components/Camera.h"

#include "../graphics/GizmoRenderer.h"

namespace PhysicsEngine
{
class GizmoSystem
{
  private:
    Guid mGuid;
    Id mId;
    World* mWorld;

    GizmoRenderer mGizmoRenderer;

  public:
    HideFlag mHide;
    bool mEnabled;

  public:
    GizmoSystem(World *world, const Id &id);
    GizmoSystem(World *world, const Guid &guid, const Id &id);
    ~GizmoSystem();

    void serialize(YAML::Node &out) const;
    void deserialize(const YAML::Node &in);

    int getType() const;
    std::string getObjectName() const;

    Guid getGuid() const;
    Id getId() const;

    void init(World *world);
    void update(const Input &input, const Time &time);

    void addToDrawList(const Line &line, const Color &color);
    void addToDrawList(const Ray &ray, float t, const Color &color);
    void addToDrawList(const Sphere &sphere, const Color &color);
    void addToDrawList(const AABB &aabb, const Color &color, bool wireframe = false);
    void addToDrawList(const Frustum &frustum, const Color &color, bool wireframe = false);
    void addToDrawList(const Plane &plane, const glm::vec3 &extents, const Color &color, bool wireframe = false);

    void clearDrawList();
};

} // namespace PhysicsEngine

#endif