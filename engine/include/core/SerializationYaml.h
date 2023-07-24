#ifndef SERIALIZATION_YAML_H__
#define SERIALIZATION_YAML_H__

#include "yaml-cpp/yaml.h"

#include "SerializationEnums.h"

namespace YAML
{
// HideFlag
template <> struct convert<PhysicsEngine::HideFlag>
{
    static Node encode(const PhysicsEngine::HideFlag &rhs)
    {
        Node node;
        node = static_cast<int>(rhs);
        return node;
    }

    static bool decode(const Node &node, PhysicsEngine::HideFlag &rhs)
    {
        rhs = static_cast<PhysicsEngine::HideFlag>(node.as<int>());
        return true;
    }
};

}

#endif