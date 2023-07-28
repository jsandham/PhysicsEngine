#ifndef COMPONENT_TYPES_H__
#define COMPONENT_TYPES_H__

#include "../core/Types.h"

#include "BoxCollider.h"
#include "Camera.h"
#include "CapsuleCollider.h"
#include "Light.h"
#include "LineRenderer.h"
#include "MeshCollider.h"
#include "MeshRenderer.h"
#include "Rigidbody.h"
#include "SphereCollider.h"
#include "Terrain.h"
#include "Transform.h"

namespace PhysicsEngine
{
    template <typename T> struct IsComponent
    {
        static constexpr bool value = false;
    };

#define INSTANTIATE(T)                          \
    template <> struct IsComponent<T>           \
    {                                           \
        static constexpr bool value = true;     \
    };

    INSTANTIATE(Transform);
    INSTANTIATE(Terrain);
    INSTANTIATE(Rigidbody);
    INSTANTIATE(MeshRenderer);
    INSTANTIATE(LineRenderer);
    INSTANTIATE(Light);
    INSTANTIATE(Camera);
    INSTANTIATE(SphereCollider);
    INSTANTIATE(MeshCollider);
    INSTANTIATE(CapsuleCollider);
    INSTANTIATE(BoxCollider);

#undef INSTANTIATE

    template <typename T> struct ComponentType
    {
        static constexpr int type = PhysicsEngine::INVALID_TYPE;
    };

    template <> struct ComponentType<Transform>
    {
        static constexpr int type = PhysicsEngine::TRANSFORM_TYPE;
    };

    template <> struct ComponentType<Terrain>
    {
        static constexpr int type = PhysicsEngine::TERRAIN_TYPE;
    };

    template <> struct ComponentType<Rigidbody>
    {
        static constexpr int type = PhysicsEngine::RIGIDBODY_TYPE;
    };

    template <> struct ComponentType<MeshRenderer>
    {
        static constexpr int type = PhysicsEngine::MESHRENDERER_TYPE;
    };

    template <> struct ComponentType<LineRenderer>
    {
        static constexpr int type = PhysicsEngine::LINERENDERER_TYPE;
    };

    template <> struct ComponentType<Light>
    {
        static constexpr int type = PhysicsEngine::LIGHT_TYPE;
    };

    template <> struct ComponentType<Camera>
    {
        static constexpr int type = CAMERA_TYPE;
    };

    template <> struct ComponentType<SphereCollider>
    {
        static constexpr int type = PhysicsEngine::SPHERECOLLIDER_TYPE;
    };

    template <> struct ComponentType<MeshCollider>
    {
        static constexpr int type = PhysicsEngine::MESHCOLLIDER_TYPE;
    };

    template <> struct ComponentType<CapsuleCollider>
    {
        static constexpr int type = CAPSULECOLLIDER_TYPE;
    };

    template <> struct ComponentType<BoxCollider>
    {
        static constexpr int type = BOXCOLLIDER_TYPE;
    };
}

#endif