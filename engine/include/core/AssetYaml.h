#ifndef ASSET_YAML_H__
#define ASSET_YAML_H__

#include <unordered_map>

#include "yaml-cpp/yaml.h"

#include "glm.h"
#include "CoreYaml.h"
#include "AssetEnums.h"
#include "Guid.h"
#include "Color.h"

namespace YAML
{
    // TextureDimension
    template <> struct convert<PhysicsEngine::TextureDimension>
    {
        static Node encode(const PhysicsEngine::TextureDimension &rhs)
        {
            Node node;
            node = static_cast<int>(rhs);
            return node;
        }

        static bool decode(const Node &node, PhysicsEngine::TextureDimension &rhs)
        {
            rhs = static_cast<PhysicsEngine::TextureDimension>(node.as<int>());
            return true;
        }
    };

    // TextureFormat
    template <> struct convert<PhysicsEngine::TextureFormat>
    {
        static Node encode(const PhysicsEngine::TextureFormat &rhs)
        {
            Node node;
            node = static_cast<int>(rhs);
            return node;
        }

        static bool decode(const Node &node, PhysicsEngine::TextureFormat &rhs)
        {
            rhs = static_cast<PhysicsEngine::TextureFormat>(node.as<int>());
            return true;
        }
    };

    // TextureWrapMode
    template <> struct convert<PhysicsEngine::TextureWrapMode>
    {
        static Node encode(const PhysicsEngine::TextureWrapMode &rhs)
        {
            Node node;
            node = static_cast<int>(rhs);
            return node;
        }

        static bool decode(const Node &node, PhysicsEngine::TextureWrapMode &rhs)
        {
            rhs = static_cast<PhysicsEngine::TextureWrapMode>(node.as<int>());
            return true;
        }
    };

    // TextureFilterMode
    template <> struct convert<PhysicsEngine::TextureFilterMode>
    {
        static Node encode(const PhysicsEngine::TextureFilterMode &rhs)
        {
            Node node;
            node = static_cast<int>(rhs);
            return node;
        }

        static bool decode(const Node &node, PhysicsEngine::TextureFilterMode &rhs)
        {
            rhs = static_cast<PhysicsEngine::TextureFilterMode>(node.as<int>());
            return true;
        }
    };

    // RenderQueue
    template <> struct convert<PhysicsEngine::RenderQueue>
    {
        static Node encode(const PhysicsEngine::RenderQueue &rhs)
        {
            Node node;
            switch (rhs)
            {
            case PhysicsEngine::RenderQueue::Opaque:
                node = "Opaque";
                break;
            case PhysicsEngine::RenderQueue::Transparent:
                node = "Transparent";
                break;
            }

            return node;
        }

        static bool decode(const Node &node, PhysicsEngine::RenderQueue &rhs)
        {
            std::string type = node.as<std::string>();
            if (type == "Opaque")
            {
                rhs = PhysicsEngine::RenderQueue::Opaque;
            }
            else if (type == "Transparent")
            {
                rhs = PhysicsEngine::RenderQueue::Transparent;
            }

            return true;
        }
    };

    // ShaderUniformType
    template <> struct convert<PhysicsEngine::ShaderUniformType>
    {
        static Node encode(const PhysicsEngine::ShaderUniformType &rhs)
        {
            Node node;

            switch (rhs)
            {
            case PhysicsEngine::ShaderUniformType::Int:
                node = "Int";
                break;
            case PhysicsEngine::ShaderUniformType::Float:
                node = "Float";
                break;
            case PhysicsEngine::ShaderUniformType::Color:
                node = "Color";
                break;
            case PhysicsEngine::ShaderUniformType::Vec2:
                node = "Vec2";
                break;
            case PhysicsEngine::ShaderUniformType::Vec3:
                node = "Vec3";
                break;
            case PhysicsEngine::ShaderUniformType::Vec4:
                node = "Vec4";
                break;
            case PhysicsEngine::ShaderUniformType::Mat2:
                node = "Mat2";
                break;
            case PhysicsEngine::ShaderUniformType::Mat3:
                node = "Mat3";
                break;
            case PhysicsEngine::ShaderUniformType::Mat4:
                node = "Mat4";
                break;
            case PhysicsEngine::ShaderUniformType::Sampler2D:
                node = "Sampler2D";
                break;
            case PhysicsEngine::ShaderUniformType::SamplerCube:
                node = "SamplerCube";
                break;
            default:
                node = "Invalid";
                break;
            }

            return node;
        }

        static bool decode(const Node &node, PhysicsEngine::ShaderUniformType &rhs)
        {
            std::string type = node.as<std::string>();
            if (type == "Int")
            {
                rhs = PhysicsEngine::ShaderUniformType::Int;
            }
            else if (type == "Float")
            {
                rhs = PhysicsEngine::ShaderUniformType::Float;
            }
            else if (type == "Color")
            {
                rhs = PhysicsEngine::ShaderUniformType::Color;
            }
            else if (type == "Vec2")
            {
                rhs = PhysicsEngine::ShaderUniformType::Vec2;
            }
            else if (type == "Vec3")
            {
                rhs = PhysicsEngine::ShaderUniformType::Vec3;
            }
            else if (type == "Vec4")
            {
                rhs = PhysicsEngine::ShaderUniformType::Vec4;
            }
            else if (type == "Mat2")
            {
                rhs = PhysicsEngine::ShaderUniformType::Mat2;
            }
            else if (type == "Mat3")
            {
                rhs = PhysicsEngine::ShaderUniformType::Mat3;
            }
            else if (type == "Mat4")
            {
                rhs = PhysicsEngine::ShaderUniformType::Mat4;
            }
            else if (type == "Sampler2D")
            {
                rhs = PhysicsEngine::ShaderUniformType::Sampler2D;
            }
            else if (type == "SamplerCube")
            {
                rhs = PhysicsEngine::ShaderUniformType::SamplerCube;
            }
            else
            {
                rhs = PhysicsEngine::ShaderUniformType::Invalid;
            }

            return true;
        }
    };

    // ShaderMacro
    template <> struct convert<PhysicsEngine::ShaderMacro>
    {
        static Node encode(const PhysicsEngine::ShaderMacro &rhs)
        {
            Node node;

            switch (rhs)
            {
            case PhysicsEngine::ShaderMacro::Directional:
                node = "Directional";
                break;
            case PhysicsEngine::ShaderMacro::Spot:
                node = "Spot";
                break;
            case PhysicsEngine::ShaderMacro::Point:
                node = "Point";
                break;
            case PhysicsEngine::ShaderMacro::HardShadows:
                node = "HardShadows";
                break;
            case PhysicsEngine::ShaderMacro::SoftShadows:
                node = "SoftShadows";
                break;
            case PhysicsEngine::ShaderMacro::SSAO:
                node = "SSAO";
                break;
            case PhysicsEngine::ShaderMacro::ShowCascades:
                node = "ShowCascades";
                break;
            case PhysicsEngine::ShaderMacro::Instancing:
                node = "Instancing";
                break;
            default:
                node = "None";
                break;
            }

            return node;
        }

        static bool decode(const Node &node, PhysicsEngine::ShaderMacro &rhs)
        {
            std::string type = node.as<std::string>();
            if (type == "Directional")
            {
                rhs = PhysicsEngine::ShaderMacro::Directional;
            }
            else if (type == "Spot")
            {
                rhs = PhysicsEngine::ShaderMacro::Spot;
            }
            else if (type == "Point")
            {
                rhs = PhysicsEngine::ShaderMacro::Point;
            }
            else if (type == "HardShadows")
            {
                rhs = PhysicsEngine::ShaderMacro::HardShadows;
            }
            else if (type == "SoftShadows")
            {
                rhs = PhysicsEngine::ShaderMacro::SoftShadows;
            }
            else if (type == "SSAO")
            {
                rhs = PhysicsEngine::ShaderMacro::SSAO;
            }
            else if (type == "ShowCascades")
            {
                rhs = PhysicsEngine::ShaderMacro::ShowCascades;
            }
            else if (type == "Instancing")
            {
                rhs = PhysicsEngine::ShaderMacro::Instancing;
            }
            else
            {
                rhs = PhysicsEngine::ShaderMacro::None;
            }

            return true;
        }
    };

    // ShaderUniform
    template <> struct convert<PhysicsEngine::ShaderUniform>
    {
        static Node encode(const PhysicsEngine::ShaderUniform &rhs)
        {
            Node node;

            node["type"] = rhs.mType;

            switch (rhs.mType)
            {
            case PhysicsEngine::ShaderUniformType::Int: {
                node["data"] = *reinterpret_cast<const int *>(rhs.mData);
                break;
            }
            case PhysicsEngine::ShaderUniformType::Float: {
                node["data"] = *reinterpret_cast<const float *>(rhs.mData);
                break;
            }
            case PhysicsEngine::ShaderUniformType::Color: {
                node["data"] = *reinterpret_cast<const PhysicsEngine::Color *>(rhs.mData);
                break;
            }
            case PhysicsEngine::ShaderUniformType::Vec2: {
                node["data"] = *reinterpret_cast<const glm::vec2 *>(rhs.mData);
                break;
            }
            case PhysicsEngine::ShaderUniformType::Vec3: {
                node["data"] = *reinterpret_cast<const glm::vec3 *>(rhs.mData);
                break;
            }
            case PhysicsEngine::ShaderUniformType::Vec4: {
                node["data"] = *reinterpret_cast<const glm::vec4 *>(rhs.mData);
                break;
            }
            case PhysicsEngine::ShaderUniformType::Sampler2D: {
                node["data"] = *reinterpret_cast<const PhysicsEngine::Guid *>(rhs.mData);
                break;
            }
            case PhysicsEngine::ShaderUniformType::SamplerCube: {
                node["data"] = *reinterpret_cast<const PhysicsEngine::Guid *>(rhs.mData);
                break;
            }
            }

            return node;
        }

        static bool decode(const Node &node, PhysicsEngine::ShaderUniform &rhs)
        {
            rhs.mType = YAML::getValue<PhysicsEngine::ShaderUniformType>(node, "type");

            switch (rhs.mType)
            {
            case PhysicsEngine::ShaderUniformType::Int: {
                int data = YAML::getValue<int>(node, "data");
                memcpy(rhs.mData, &data, sizeof(int));
                break;
            }
            case PhysicsEngine::ShaderUniformType::Float: {
                float data = YAML::getValue<float>(node, "data");
                memcpy(rhs.mData, &data, sizeof(float));
                break;
            }
            case PhysicsEngine::ShaderUniformType::Color: {
                PhysicsEngine::Color data = YAML::getValue<PhysicsEngine::Color>(node, "data");
                memcpy(rhs.mData, &data, sizeof(PhysicsEngine::Color));
                break;
            }
            case PhysicsEngine::ShaderUniformType::Vec2: {
                glm::vec2 data = YAML::getValue<glm::vec2>(node, "data");
                memcpy(rhs.mData, &data, sizeof(glm::vec2));
                break;
            }
            case PhysicsEngine::ShaderUniformType::Vec3: {
                glm::vec3 data = YAML::getValue<glm::vec3>(node, "data");
                memcpy(rhs.mData, &data, sizeof(glm::vec3));
                break;
            }
            case PhysicsEngine::ShaderUniformType::Vec4: {
                glm::vec4 data = YAML::getValue<glm::vec4>(node, "data");
                memcpy(rhs.mData, &data, sizeof(glm::vec4));
                break;
            }
            case PhysicsEngine::ShaderUniformType::Sampler2D: {
                PhysicsEngine::Guid data = YAML::getValue<PhysicsEngine::Guid>(node, "data");
                memcpy(rhs.mData, &data, sizeof(PhysicsEngine::Guid));
                break;
            }
            case PhysicsEngine::ShaderUniformType::SamplerCube: {
                PhysicsEngine::Guid data = YAML::getValue<PhysicsEngine::Guid>(node, "data");
                memcpy(rhs.mData, &data, sizeof(PhysicsEngine::Guid));
                break;
            }
            }

            return true;
        }
    };

    // std::set<ShaderMacro>
    template <> struct convert<std::set<PhysicsEngine::ShaderMacro>>
    {
        static Node encode(const std::set<PhysicsEngine::ShaderMacro> &rhs)
        {
            Node node = YAML::Load("[]");

            for (auto it = rhs.begin(); it != rhs.end(); it++)
            {
                node.push_back(*it);
            }

            return node;
        }

        static bool decode(const Node &node, std::set<PhysicsEngine::ShaderMacro> &rhs)
        {
            if (!node.IsSequence())
            {
                return false;
            }

            for (YAML::const_iterator it = node.begin(); it != node.end(); ++it)
            {
                rhs.insert(it->as<PhysicsEngine::ShaderMacro>());
            }

            return true;
        }
    };

    // std::unordered_map<int, std::set<ShaderMacro>>
    template <> struct convert<std::unordered_map<int, std::set<PhysicsEngine::ShaderMacro>>>
    {
        static Node encode(const std::unordered_map<int, std::set<PhysicsEngine::ShaderMacro>> &rhs)
        {
            Node node;

            for (auto it = rhs.begin(); it != rhs.end(); it++)
            {
                node[std::to_string(it->first)] = it->second;
            }

            return node;
        }

        static bool decode(const Node &node, std::unordered_map<int, std::set<PhysicsEngine::ShaderMacro>> &rhs)
        {
            if (!node.IsMap())
            {
                return false;
            }

            for (YAML::const_iterator it = node.begin(); it != node.end(); ++it)
            {
                rhs[it->first.as<int>()] = it->second.as<std::set<PhysicsEngine::ShaderMacro>>();
            }

            return true;
        }
    };
} // namespace YAML

#endif