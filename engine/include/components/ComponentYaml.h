#ifndef COMPONENT_YAML_H__
#define COMPONENT_YAML_H__

#include "yaml-cpp/yaml.h"

#include "ComponentEnums.h"

namespace YAML
{

// LightType
template <> struct convert<PhysicsEngine::LightType>
{
    static Node encode(const PhysicsEngine::LightType &rhs)
    {
        Node node;
        node = static_cast<int>(rhs);
        return node;
    }

    static bool decode(const Node &node, PhysicsEngine::LightType &rhs)
    {
        rhs = static_cast<PhysicsEngine::LightType>(node.as<int>());
        return true;
    }
};

// ShadowType
template <> struct convert<PhysicsEngine::ShadowType>
{
    static Node encode(const PhysicsEngine::ShadowType &rhs)
    {
        Node node;
        node = static_cast<int>(rhs);
        return node;
    }

    static bool decode(const Node &node, PhysicsEngine::ShadowType &rhs)
    {
        rhs = static_cast<PhysicsEngine::ShadowType>(node.as<int>());
        return true;
    }
};

// ShadowMapResolution
template <> struct convert<PhysicsEngine::ShadowMapResolution>
{
    static Node encode(const PhysicsEngine::ShadowMapResolution &rhs)
    {
        Node node;
        node = static_cast<int>(rhs);
        return node;
    }

    static bool decode(const Node &node, PhysicsEngine::ShadowMapResolution &rhs)
    {
        rhs = static_cast<PhysicsEngine::ShadowMapResolution>(node.as<int>());
        return true;
    }
};

// CameraMode
template <> struct convert<PhysicsEngine::CameraMode>
{
    static Node encode(const PhysicsEngine::CameraMode &rhs)
    {
        Node node;
        node = static_cast<int>(rhs);
        return node;
    }

    static bool decode(const Node &node, PhysicsEngine::CameraMode &rhs)
    {
        rhs = static_cast<PhysicsEngine::CameraMode>(node.as<int>());
        return true;
    }
};

// CameraSSAO
template <> struct convert<PhysicsEngine::CameraSSAO>
{
    static Node encode(const PhysicsEngine::CameraSSAO &rhs)
    {
        Node node;
        node = static_cast<int>(rhs);
        return node;
    }

    static bool decode(const Node &node, PhysicsEngine::CameraSSAO &rhs)
    {
        rhs = static_cast<PhysicsEngine::CameraSSAO>(node.as<int>());
        return true;
    }
};

// CameraGizmos
template <> struct convert<PhysicsEngine::CameraGizmos>
{
    static Node encode(const PhysicsEngine::CameraGizmos &rhs)
    {
        Node node;
        node = static_cast<int>(rhs);
        return node;
    }

    static bool decode(const Node &node, PhysicsEngine::CameraGizmos &rhs)
    {
        rhs = static_cast<PhysicsEngine::CameraGizmos>(node.as<int>());
        return true;
    }
};

// RenderPath
template <> struct convert<PhysicsEngine::RenderPath>
{
    static Node encode(const PhysicsEngine::RenderPath &rhs)
    {
        Node node;
        node = static_cast<int>(rhs);
        return node;
    }

    static bool decode(const Node &node, PhysicsEngine::RenderPath &rhs)
    {
        rhs = static_cast<PhysicsEngine::RenderPath>(node.as<int>());
        return true;
    }
};

// ColorTarget
template <> struct convert<PhysicsEngine::ColorTarget>
{
    static Node encode(const PhysicsEngine::ColorTarget &rhs)
    {
        Node node;
        node = static_cast<int>(rhs);
        return node;
    }

    static bool decode(const Node &node, PhysicsEngine::ColorTarget &rhs)
    {
        rhs = static_cast<PhysicsEngine::ColorTarget>(node.as<int>());
        return true;
    }
};

// ShadowCascades
template <> struct convert<PhysicsEngine::ShadowCascades>
{
    static Node encode(const PhysicsEngine::ShadowCascades &rhs)
    {
        Node node;
        node = static_cast<int>(rhs);
        return node;
    }

    static bool decode(const Node &node, PhysicsEngine::ShadowCascades &rhs)
    {
        rhs = static_cast<PhysicsEngine::ShadowCascades>(node.as<int>());
        return true;
    }
};

} // namespace YAML

#endif