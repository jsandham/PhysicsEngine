#ifndef GIZMOSYSTEM_H__
#define GIZMOSYSTEM_H__

#include <vector>

#include "System.h"

#include "../core/AABB.h"
#include "../core/Color.h"
#include "../core/Frustum.h"
#include "../core/Input.h"
#include "../core/Line.h"
#include "../core/Plane.h"
#include "../core/Ray.h"
#include "../core/Sphere.h"

#include "../components/Camera.h"

#include "../graphics/GizmoRenderer.h"

namespace PhysicsEngine
{
class GizmoSystem : public System
{
  private:
    GizmoRenderer mGizmoRenderer;

  public:
    GizmoSystem(World* world);
    GizmoSystem(World* world, Guid id);
    ~GizmoSystem();

    virtual void serialize(YAML::Node &out) const override;
    virtual void deserialize(const YAML::Node &in) override;

    virtual int getType() const override;
    virtual std::string getObjectName() const override;

    void init(World *world) override;
    void update(const Input &input, const Time &time) override;

    void addToDrawList(const Line &line, const Color &color);
    void addToDrawList(const Ray &ray, float t, const Color &color);
    void addToDrawList(const Sphere& sphere, const Color& color);
    void addToDrawList(const AABB &aabb, const Color &color, bool wireframe = false);
    void addToDrawList(const Frustum &frustum, const Color &color, bool wireframe = false);
    void addToDrawList(const Plane &plane, const glm::vec3 &extents, const Color &color, bool wireframe = false);

    void clearDrawList();
};

template <> struct SystemType<GizmoSystem>
{
    static constexpr int type = PhysicsEngine::GIZMOSYSTEM_TYPE;
};
template <> struct IsSystemInternal<GizmoSystem>
{
    static constexpr bool value = true;
};
} // namespace PhysicsEngine

#endif