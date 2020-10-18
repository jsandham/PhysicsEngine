#ifndef __GIZMOSYSTEM_H__
#define __GIZMOSYSTEM_H__

#include <vector>

#include "System.h"

#include "../core/Input.h"
#include "../core/Color.h"
#include "../core/Line.h"
#include "../core/Ray.h"
#include "../core/AABB.h"
#include "../core/Sphere.h"
#include "../core/Frustum.h"
#include "../core/Plane.h"

#include "../components/Camera.h"

#include "../graphics/GizmoRenderer.h"

namespace PhysicsEngine
{
#pragma pack(push, 1)
    struct GizmoSystemHeader
    {
        Guid mSystemId;
        int32_t mUpdateOrder;
    };
#pragma pack(pop)

    class GizmoSystem : public System
    {
    private:
        GizmoRenderer mGizmoRenderer;

    public:
        GizmoSystem();
        GizmoSystem(const std::vector<char>& data);
        ~GizmoSystem();

        std::vector<char> serialize() const;
        std::vector<char> serialize(const Guid &systemId) const;
        void deserialize(const std::vector<char>& data);

        void init(World* world);
        void update(const Input& input, const Time& time);

        void addToDrawList(const Line &line, const Color &color);
        void addToDrawList(const Ray& ray, float t, const Color& color);
        void addToDrawList(const AABB& aabb, const Color& color);
        void addToDrawList(const Sphere& sphere, const Color& color);
        void addToDrawList(const Frustum& frustum, const Color& color);
        void addToDrawList(const Plane& plane, const glm::vec3 &extents, const Color& color);

        void clearDrawList();
    };

    template <typename T> struct IsGizmoSystem
    {
        static constexpr bool value = false;
    };

    template <> struct SystemType<GizmoSystem>
    {
        static constexpr int type = PhysicsEngine::GIZMOSYSTEM_TYPE;
    };
    template <> struct IsGizmoSystem<GizmoSystem>
    {
        static constexpr bool value = true;
    };
    template <> struct IsSystem<GizmoSystem>
    {
        static constexpr bool value = true;
    };
    template <> struct IsSystemInternal<GizmoSystem>
    {
        static constexpr bool value = true;
    };
} // namespace PhysicsEngine

#endif