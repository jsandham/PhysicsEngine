#ifndef YAML_EXTENSIONS_H__
#define YAML_EXTENSIONS_H__

#include "yaml-cpp/yaml.h"

#include "../glm/glm.hpp"
#include "../glm/gtx/quaternion.hpp"

#include "Guid.h"
#include "Color.h"
#include "AABB.h"
#include "Capsule.h"
#include "Sphere.h"
#include "Circle.h"
#include "Line.h"
#include "Plane.h"
#include "Ray.h"
#include "Triangle.h"
#include "Frustum.h"
#include "Viewport.h"

namespace PhysicsEngine
{
    // Forward enum declarations
    enum class LightType : int;
    enum class ShadowType : int;
    enum class ShadowMapResolution : int;

    enum class CameraMode : int;
    enum class CameraSSAO : int;
    enum class CameraGizmos : int;
    enum class RenderPath : int;

    enum class RenderQueue : int;

    enum class TextureDimension : int;
    enum class TextureFormat : int;
    enum class TextureWrapMode : int;
    enum class TextureFilterMode : int;
}

namespace YAML {
    // LightType
    template<>
    struct convert<PhysicsEngine::LightType> {
        static Node encode(const PhysicsEngine::LightType& rhs) {
            Node node;
            node = static_cast<int>(rhs);
            return node;
        }

        static bool decode(const Node& node, PhysicsEngine::LightType& rhs) {
            rhs = static_cast<PhysicsEngine::LightType>(node.as<int>());
            return true;
        }
    };

    // ShadowType
    template<>
    struct convert<PhysicsEngine::ShadowType> {
        static Node encode(const PhysicsEngine::ShadowType& rhs) {
            Node node;
            node = static_cast<int>(rhs);
            return node;
        }

        static bool decode(const Node& node, PhysicsEngine::ShadowType& rhs) {
            rhs = static_cast<PhysicsEngine::ShadowType>(node.as<int>());
            return true;
        }
    };

    // ShadowMapResolution
    template<>
    struct convert<PhysicsEngine::ShadowMapResolution> {
        static Node encode(const PhysicsEngine::ShadowMapResolution& rhs) {
            Node node;
            node = static_cast<int>(rhs);
            return node;
        }

        static bool decode(const Node& node, PhysicsEngine::ShadowMapResolution& rhs) {
            rhs = static_cast<PhysicsEngine::ShadowMapResolution>(node.as<int>());
            return true;
        }
    };

    // CameraMode
    template<>
    struct convert<PhysicsEngine::CameraMode> {
        static Node encode(const PhysicsEngine::CameraMode& rhs) {
            Node node;
            node = static_cast<int>(rhs);
            return node;
        }

        static bool decode(const Node& node, PhysicsEngine::CameraMode& rhs) {
            rhs = static_cast<PhysicsEngine::CameraMode>(node.as<int>());
            return true;
        }
    };

    // CameraSSAO
    template<>
    struct convert<PhysicsEngine::CameraSSAO> {
        static Node encode(const PhysicsEngine::CameraSSAO& rhs) {
            Node node;
            node = static_cast<int>(rhs);
            return node;
        }

        static bool decode(const Node& node, PhysicsEngine::CameraSSAO& rhs) {
            rhs = static_cast<PhysicsEngine::CameraSSAO>(node.as<int>());
            return true;
        }
    };

    // CameraGizmos
    template<>
    struct convert<PhysicsEngine::CameraGizmos> {
        static Node encode(const PhysicsEngine::CameraGizmos& rhs) {
            Node node;
            node = static_cast<int>(rhs);
            return node;
        }

        static bool decode(const Node& node, PhysicsEngine::CameraGizmos& rhs) {
            rhs = static_cast<PhysicsEngine::CameraGizmos>(node.as<int>());
            return true;
        }
    };

    // RenderPath
    template<>
    struct convert<PhysicsEngine::RenderPath> {
        static Node encode(const PhysicsEngine::RenderPath& rhs) {
            Node node;
            node = static_cast<int>(rhs);
            return node;
        }

        static bool decode(const Node& node, PhysicsEngine::RenderPath& rhs) {
            rhs = static_cast<PhysicsEngine::RenderPath>(node.as<int>());
            return true;
        }
    };

    // RenderQueue
    template<>
    struct convert<PhysicsEngine::RenderQueue> {
        static Node encode(const PhysicsEngine::RenderQueue& rhs) {
            Node node;
            node = static_cast<int>(rhs);
            return node;
        }

        static bool decode(const Node& node, PhysicsEngine::RenderQueue& rhs) {
            rhs = static_cast<PhysicsEngine::RenderQueue>(node.as<int>());
            return true;
        }
    };

    // TextureDimension
    template<>
    struct convert<PhysicsEngine::TextureDimension> {
        static Node encode(const PhysicsEngine::TextureDimension& rhs) {
            Node node;
            node = static_cast<int>(rhs);
            return node;
        }

        static bool decode(const Node& node, PhysicsEngine::TextureDimension& rhs) {
            rhs = static_cast<PhysicsEngine::TextureDimension>(node.as<int>());
            return true;
        }
    };

    // TextureFormat
    template<>
    struct convert<PhysicsEngine::TextureFormat> {
        static Node encode(const PhysicsEngine::TextureFormat& rhs) {
            Node node;
            node = static_cast<int>(rhs);
            return node;
        }

        static bool decode(const Node& node, PhysicsEngine::TextureFormat& rhs) {
            rhs = static_cast<PhysicsEngine::TextureFormat>(node.as<int>());
            return true;
        }
    };

    // TextureWrapMode
    template<>
    struct convert<PhysicsEngine::TextureWrapMode> {
        static Node encode(const PhysicsEngine::TextureWrapMode& rhs) {
            Node node;
            node = static_cast<int>(rhs);
            return node;
        }

        static bool decode(const Node& node, PhysicsEngine::TextureWrapMode& rhs) {
            rhs = static_cast<PhysicsEngine::TextureWrapMode>(node.as<int>());
            return true;
        }
    };

    // TextureFilterMode
    template<>
    struct convert<PhysicsEngine::TextureFilterMode> {
        static Node encode(const PhysicsEngine::TextureFilterMode& rhs) {
            Node node;
            node = static_cast<int>(rhs);
            return node;
        }

        static bool decode(const Node& node, PhysicsEngine::TextureFilterMode& rhs) {
            rhs = static_cast<PhysicsEngine::TextureFilterMode>(node.as<int>());
            return true;
        }
    };

    // Guid
    template<>
    struct convert<PhysicsEngine::Guid> {
        static Node encode(const PhysicsEngine::Guid& rhs) {
            Node node;
            node = rhs.toString();
            return node;
        }

        static bool decode(const Node& node, PhysicsEngine::Guid& rhs) {
            rhs = node.as<std::string>();
            return true;
        }
    };

    // Color
    template<>
    struct convert<PhysicsEngine::Color> {
        static Node encode(const PhysicsEngine::Color& rhs) {
            Node node;
            node.push_back(rhs.r);
            node.push_back(rhs.g);
            node.push_back(rhs.b);
            node.push_back(rhs.a);
            return node;
        }

        static bool decode(const Node& node, PhysicsEngine::Color& rhs) {
            if (!node.IsSequence() || node.size() != 4) {
                return false;
            }

            rhs.r = node[0].as<float>();
            rhs.g = node[1].as<float>();
            rhs.b = node[2].as<float>();
            rhs.a = node[3].as<float>();
            return true;
        }
    };

    // Color32
    template<>
    struct convert<PhysicsEngine::Color32> {
        static Node encode(const PhysicsEngine::Color32& rhs) {
            Node node;
            node.push_back(rhs.r);
            node.push_back(rhs.g);
            node.push_back(rhs.b);
            node.push_back(rhs.a);
            return node;
        }

        static bool decode(const Node& node, PhysicsEngine::Color32& rhs) {
            if (!node.IsSequence() || node.size() != 4) {
                return false;
            }

            rhs.r = node[0].as<unsigned char>();
            rhs.g = node[1].as<unsigned char>();
            rhs.b = node[2].as<unsigned char>();
            rhs.a = node[3].as<unsigned char>();
            return true;
        }
    };

    // AABB
    template<>
    struct convert<PhysicsEngine::AABB> {
        static Node encode(const PhysicsEngine::AABB& rhs) {
            Node node;
            node["centre"] = rhs.mCentre;
            node["size"] = rhs.mSize;
            return node;
        }

        static bool decode(const Node& node, PhysicsEngine::AABB& rhs) {
            rhs.mCentre = node["centre"].as<glm::vec3>();
            rhs.mSize = node["size"].as<glm::vec3>();
            return true;
        }
    };

    // Capsule
    template<>
    struct convert<PhysicsEngine::Capsule> {
        static Node encode(const PhysicsEngine::Capsule& rhs) {
            Node node;
            node["centre"] = rhs.mCentre;
            node["radius"] = rhs.mRadius;
            node["height"] = rhs.mHeight;
            return node;
        }

        static bool decode(const Node& node, PhysicsEngine::Capsule& rhs) {
            rhs.mCentre = node["centre"].as<glm::vec3>();
            rhs.mRadius = node["radius"].as<float>();
            rhs.mHeight = node["height"].as<float>();
            return true;
        }
    };

    // Sphere
    template<>
    struct convert<PhysicsEngine::Sphere> {
        static Node encode(const PhysicsEngine::Sphere& rhs) {
            Node node;
            node["centre"] = rhs.mCentre;
            node["radius"] = rhs.mRadius;
            return node;
        }

        static bool decode(const Node& node, PhysicsEngine::Sphere& rhs) {
            rhs.mCentre = node["centre"].as<glm::vec3>();
            rhs.mRadius = node["radius"].as<float>();
            return true;
        }
    };

    // Circle
    template<>
    struct convert<PhysicsEngine::Circle> {
        static Node encode(const PhysicsEngine::Circle& rhs) {
            Node node;
            node["centre"] = rhs.mCentre;
            node["normal"] = rhs.mNormal;
            node["radius"] = rhs.mRadius;
            return node;
        }

        static bool decode(const Node& node, PhysicsEngine::Circle& rhs) {
            rhs.mCentre = node["centre"].as<glm::vec3>();
            rhs.mNormal = node["normal"].as<glm::vec3>();
            rhs.mRadius = node["radius"].as<float>();

            return true;
        }
    };

    // Line
    template<>
    struct convert<PhysicsEngine::Line> {
        static Node encode(const PhysicsEngine::Line& rhs) {
            Node node;
            node["start"] = rhs.mStart;
            node["end"] = rhs.mEnd;
            return node;
        }

        static bool decode(const Node& node, PhysicsEngine::Line& rhs) {
            rhs.mStart = node["start"].as<glm::vec3>();
            rhs.mEnd = node["end"].as<glm::vec3>();

            return true;
        }
    };

    // Plane
    template<>
    struct convert<PhysicsEngine::Plane> {
        static Node encode(const PhysicsEngine::Plane& rhs) {
            Node node;
            node["normal"] = rhs.mNormal;
            node["x0"] = rhs.mX0;
            return node;
        }

        static bool decode(const Node& node, PhysicsEngine::Plane& rhs) {
            rhs.mNormal = node["normal"].as<glm::vec3>();
            rhs.mX0 = node["x0"].as<glm::vec3>();

            return true;
        }
    };

    // Ray
    template<>
    struct convert<PhysicsEngine::Ray> {
        static Node encode(const PhysicsEngine::Ray& rhs) {
            Node node;
            node["origin"] = rhs.mOrigin;
            node["direction"] = rhs.mDirection;
            return node;
        }

        static bool decode(const Node& node, PhysicsEngine::Ray& rhs) {
            rhs.mOrigin = node["origin"].as<glm::vec3>();
            rhs.mDirection = node["direction"].as<glm::vec3>();

            return true;
        }
    };

    // Triangle
    template<>
    struct convert<PhysicsEngine::Triangle> {
        static Node encode(const PhysicsEngine::Triangle& rhs) {
            Node node;
            node["v0"] = rhs.mV0;
            node["v1"] = rhs.mV1;
            node["v2"] = rhs.mV2;
            return node;
        }

        static bool decode(const Node& node, PhysicsEngine::Triangle& rhs) {
            rhs.mV0 = node["v0"].as<glm::vec3>();
            rhs.mV1 = node["v1"].as<glm::vec3>();
            rhs.mV2 = node["v2"].as<glm::vec3>();

            return true;
        }
    };

    // Frustum
    template<>
    struct convert<PhysicsEngine::Frustum> {
        static Node encode(const PhysicsEngine::Frustum& rhs) {
            Node node;
            node["fov"] = rhs.mFov;
            node["aspectRatio"] = rhs.mAspectRatio;
            node["nearPlane"] = rhs.mNearPlane;
            node["farPlane"] = rhs.mFarPlane;
            return node;
        }

        static bool decode(const Node& node, PhysicsEngine::Frustum& rhs) {
            rhs.mFov = node["fov"].as<float>();
            rhs.mAspectRatio = node["aspectRatio"].as<float>();
            rhs.mNearPlane = node["nearPlane"].as<float>();
            rhs.mFarPlane = node["farPlane"].as<float>();
            return true;
        }
    };

    // Viewport
    template<>
    struct convert<PhysicsEngine::Viewport> {
        static Node encode(const PhysicsEngine::Viewport& rhs) {
            Node node;
            node["x"] = rhs.mX;
            node["y"] = rhs.mY;
            node["width"] = rhs.mWidth;
            node["height"] = rhs.mHeight;
            return node;
        }

        static bool decode(const Node& node, PhysicsEngine::Viewport& rhs) {
            rhs.mX = node["x"].as<int>();
            rhs.mY = node["y"].as<int>();
            rhs.mWidth = node["width"].as<int>();
            rhs.mHeight = node["height"].as<int>();
            return true;
        }
    };

    // vec2
    template<>
    struct convert<glm::vec2> {
        static Node encode(const glm::vec2& rhs) {
            Node node;
            node.push_back(rhs.x);
            node.push_back(rhs.y);
            return node;
        }

        static bool decode(const Node& node, glm::vec2& rhs) {
            if (!node.IsSequence() || node.size() != 2) {
                return false;
            }

            rhs.x = node[0].as<float>();
            rhs.y = node[1].as<float>();
            return true;
        }
    };

    // vec3
    template<>
    struct convert<glm::vec3> {
        static Node encode(const glm::vec3& rhs) {
            Node node;
            node.push_back(rhs.x);
            node.push_back(rhs.y);
            node.push_back(rhs.z);
            return node;
        }

        static bool decode(const Node& node, glm::vec3& rhs) {
            if (!node.IsSequence() || node.size() != 3) {
                return false;
            }

            rhs.x = node[0].as<float>();
            rhs.y = node[1].as<float>();
            rhs.z = node[2].as<float>();
            return true;
        }
    };

    // vec4
    template<>
    struct convert<glm::vec4> {
        static Node encode(const glm::vec4& rhs) {
            Node node;
            node.push_back(rhs.x);
            node.push_back(rhs.y);
            node.push_back(rhs.z);
            node.push_back(rhs.w);
            return node;
        }

        static bool decode(const Node& node, glm::vec4& rhs) {
            if (!node.IsSequence() || node.size() != 4) {
                return false;
            }

            rhs.x = node[0].as<float>();
            rhs.y = node[1].as<float>();
            rhs.z = node[2].as<float>();
            rhs.w = node[3].as<float>();
            return true;
        }
    };

    // quat
    template<>
    struct convert<glm::quat> {
        static Node encode(const glm::quat& rhs) {
            Node node;
            node.push_back(rhs.x);
            node.push_back(rhs.y);
            node.push_back(rhs.z);
            node.push_back(rhs.w);
            return node;
        }

        static bool decode(const Node& node, glm::quat& rhs) {
            if (!node.IsSequence() || node.size() != 4) {
                return false;
            }

            rhs.x = node[0].as<float>();
            rhs.y = node[1].as<float>();
            rhs.z = node[2].as<float>();
            rhs.w = node[3].as<float>();
            return true;
        }
    };
}

#endif