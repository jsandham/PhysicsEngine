#ifndef GLM_YAML_H__
#define GLM_YAML_H__

#include "yaml-cpp/yaml.h"

#include "glm.h"
#include "Log.h"
#include "Guid.h"
#include "Sphere.h"
#include "AABB.h"
#include "Capsule.h"
#include "Triangle.h"
#include "Plane.h"
#include "Circle.h"
#include "Line.h"
#include "Rect.h"
#include "Viewport.h"
#include "Frustum.h"
#include "Color.h"
#include "Ray.h"


namespace YAML
{
template <typename T> T getValue(const Node &node, const std::string &key)
{
    if (node[key])
    {
        return node[key].as<T>();
    }

    return T();
}

template <typename T> T getValue(const Node &node, const std::string &key, int index)
{
    if (node[key][index])
    {
        return node[key][index].as<T>();
    }

    return T();
}

template <> std::string getValue<std::string>(const Node &node, const std::string &key);

// vec2
template <> struct convert<glm::vec2>
{
    static Node encode(const glm::vec2 &rhs)
    {
        Node node;
        node.push_back(rhs.x);
        node.push_back(rhs.y);
        return node;
    }

    static bool decode(const Node &node, glm::vec2 &rhs)
    {
        if (!node.IsSequence() || node.size() != 2)
        {
            return false;
        }

        rhs.x = node[0].as<float>();
        rhs.y = node[1].as<float>();
        return true;
    }
};

// vec3
template <> struct convert<glm::vec3>
{
    static Node encode(const glm::vec3 &rhs)
    {
        Node node;
        node.push_back(rhs.x);
        node.push_back(rhs.y);
        node.push_back(rhs.z);
        return node;
    }

    static bool decode(const Node &node, glm::vec3 &rhs)
    {
        if (!node.IsSequence() || node.size() != 3)
        {
            return false;
        }

        rhs.x = node[0].as<float>();
        rhs.y = node[1].as<float>();
        rhs.z = node[2].as<float>();
        return true;
    }
};

// vec4
template <> struct convert<glm::vec4>
{
    static Node encode(const glm::vec4 &rhs)
    {
        Node node;
        node.push_back(rhs.x);
        node.push_back(rhs.y);
        node.push_back(rhs.z);
        node.push_back(rhs.w);
        return node;
    }

    static bool decode(const Node &node, glm::vec4 &rhs)
    {
        if (!node.IsSequence() || node.size() != 4)
        {
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
template <> struct convert<glm::quat>
{
    static Node encode(const glm::quat &rhs)
    {
        Node node;
        node.push_back(rhs.x);
        node.push_back(rhs.y);
        node.push_back(rhs.z);
        node.push_back(rhs.w);
        return node;
    }

    static bool decode(const Node &node, glm::quat &rhs)
    {
        if (!node.IsSequence() || node.size() != 4)
        {
            return false;
        }

        rhs.x = node[0].as<float>();
        rhs.y = node[1].as<float>();
        rhs.z = node[2].as<float>();
        rhs.w = node[3].as<float>();
        return true;
    }
};

// Guid
template <> struct convert<PhysicsEngine::Guid>
{
    static Node encode(const PhysicsEngine::Guid &rhs)
    {
        Node node;
        node = rhs.toString();
        return node;
    }

    static bool decode(const Node &node, PhysicsEngine::Guid &rhs)
    {
        rhs = node.as<std::string>();
        return true;
    }
};

// Sphere
template <> struct convert<PhysicsEngine::Sphere>
{
    static Node encode(const PhysicsEngine::Sphere &rhs)
    {
        Node node;
        node["centre"] = rhs.mCentre;
        node["radius"] = rhs.mRadius;
        return node;
    }

    static bool decode(const Node &node, PhysicsEngine::Sphere &rhs)
    {
        rhs.mCentre = node["centre"].as<glm::vec3>();
        rhs.mRadius = node["radius"].as<float>();
        return true;
    }
};

// AABB
template <> struct convert<PhysicsEngine::AABB>
{
    static Node encode(const PhysicsEngine::AABB &rhs)
    {
        Node node;
        node["centre"] = rhs.mCentre;
        node["size"] = rhs.mSize;
        return node;
    }

    static bool decode(const Node &node, PhysicsEngine::AABB &rhs)
    {
        rhs.mCentre = node["centre"].as<glm::vec3>();
        rhs.mSize = node["size"].as<glm::vec3>();
        return true;
    }
};

// Capsule
template <> struct convert<PhysicsEngine::Capsule>
{
    static Node encode(const PhysicsEngine::Capsule &rhs)
    {
        Node node;
        node["centre"] = rhs.mCentre;
        node["radius"] = rhs.mRadius;
        node["height"] = rhs.mHeight;
        return node;
    }

    static bool decode(const Node &node, PhysicsEngine::Capsule &rhs)
    {
        rhs.mCentre = node["centre"].as<glm::vec3>();
        rhs.mRadius = node["radius"].as<float>();
        rhs.mHeight = node["height"].as<float>();
        return true;
    }
};

// Triangle
template <> struct convert<PhysicsEngine::Triangle>
{
    static Node encode(const PhysicsEngine::Triangle &rhs)
    {
        Node node;
        node["v0"] = rhs.mV0;
        node["v1"] = rhs.mV1;
        node["v2"] = rhs.mV2;
        return node;
    }

    static bool decode(const Node &node, PhysicsEngine::Triangle &rhs)
    {
        rhs.mV0 = node["v0"].as<glm::vec3>();
        rhs.mV1 = node["v1"].as<glm::vec3>();
        rhs.mV2 = node["v2"].as<glm::vec3>();

        return true;
    }
};

// Circle
template <> struct convert<PhysicsEngine::Circle>
{
    static Node encode(const PhysicsEngine::Circle &rhs)
    {
        Node node;
        node["centre"] = rhs.mCentre;
        node["normal"] = rhs.mNormal;
        node["radius"] = rhs.mRadius;
        return node;
    }

    static bool decode(const Node &node, PhysicsEngine::Circle &rhs)
    {
        rhs.mCentre = node["centre"].as<glm::vec3>();
        rhs.mNormal = node["normal"].as<glm::vec3>();
        rhs.mRadius = node["radius"].as<float>();

        return true;
    }
};

// Line
template <> struct convert<PhysicsEngine::Line>
{
    static Node encode(const PhysicsEngine::Line &rhs)
    {
        Node node;
        node["start"] = rhs.mStart;
        node["end"] = rhs.mEnd;
        return node;
    }

    static bool decode(const Node &node, PhysicsEngine::Line &rhs)
    {
        rhs.mStart = node["start"].as<glm::vec3>();
        rhs.mEnd = node["end"].as<glm::vec3>();

        return true;
    }
};

// Rect
template <> struct convert<PhysicsEngine::Rect>
{
    static Node encode(const PhysicsEngine::Rect &rhs)
    {
        Node node;
        node["x"] = rhs.mX;
        node["y"] = rhs.mY;
        node["width"] = rhs.mWidth;
        node["height"] = rhs.mHeight;
        return node;
    }

    static bool decode(const Node &node, PhysicsEngine::Rect &rhs)
    {
        rhs.mX = node["x"].as<float>();
        rhs.mY = node["y"].as<float>();
        rhs.mWidth = node["width"].as<float>();
        rhs.mHeight = node["height"].as<float>();
        return true;
    }
};

// Plane
template <> struct convert<PhysicsEngine::Plane>
{
    static Node encode(const PhysicsEngine::Plane &rhs)
    {
        Node node;
        node["normal"] = rhs.mNormal;
        node["x0"] = rhs.mX0;
        return node;
    }

    static bool decode(const Node &node, PhysicsEngine::Plane &rhs)
    {
        rhs.mNormal = node["normal"].as<glm::vec3>();
        rhs.mX0 = node["x0"].as<glm::vec3>();

        return true;
    }
};

// Viewport
template <> struct convert<PhysicsEngine::Viewport>
{
    static Node encode(const PhysicsEngine::Viewport &rhs)
    {
        Node node;
        node["x"] = rhs.mX;
        node["y"] = rhs.mY;
        node["width"] = rhs.mWidth;
        node["height"] = rhs.mHeight;
        return node;
    }

    static bool decode(const Node &node, PhysicsEngine::Viewport &rhs)
    {
        rhs.mX = node["x"].as<int>();
        rhs.mY = node["y"].as<int>();
        rhs.mWidth = node["width"].as<int>();
        rhs.mHeight = node["height"].as<int>();
        return true;
    }
};

// Frustum
template <> struct convert<PhysicsEngine::Frustum>
{
    static Node encode(const PhysicsEngine::Frustum &rhs)
    {
        Node node;
        node["fov"] = rhs.mFov;
        node["aspectRatio"] = rhs.mAspectRatio;
        node["nearPlane"] = rhs.mNearPlane;
        node["farPlane"] = rhs.mFarPlane;
        return node;
    }

    static bool decode(const Node &node, PhysicsEngine::Frustum &rhs)
    {
        rhs.mFov = node["fov"].as<float>();
        rhs.mAspectRatio = node["aspectRatio"].as<float>();
        rhs.mNearPlane = node["nearPlane"].as<float>();
        rhs.mFarPlane = node["farPlane"].as<float>();
        return true;
    }
};

// Color
template <> struct convert<PhysicsEngine::Color>
{
    static Node encode(const PhysicsEngine::Color &rhs)
    {
        Node node;
        node.push_back(rhs.mR);
        node.push_back(rhs.mG);
        node.push_back(rhs.mB);
        node.push_back(rhs.mA);
        return node;
    }

    static bool decode(const Node &node, PhysicsEngine::Color &rhs)
    {
        if (!node.IsSequence() || node.size() != 4)
        {
            return false;
        }

        rhs.mR = node[0].as<float>();
        rhs.mG = node[1].as<float>();
        rhs.mB = node[2].as<float>();
        rhs.mA = node[3].as<float>();
        return true;
    }
};

// Color32
template <> struct convert<PhysicsEngine::Color32>
{
    static Node encode(const PhysicsEngine::Color32 &rhs)
    {
        Node node;
        node.push_back(rhs.mR);
        node.push_back(rhs.mG);
        node.push_back(rhs.mB);
        node.push_back(rhs.mA);
        return node;
    }

    static bool decode(const Node &node, PhysicsEngine::Color32 &rhs)
    {
        if (!node.IsSequence() || node.size() != 4)
        {
            return false;
        }

        rhs.mR = node[0].as<unsigned char>();
        rhs.mG = node[1].as<unsigned char>();
        rhs.mB = node[2].as<unsigned char>();
        rhs.mA = node[3].as<unsigned char>();
        return true;
    }
};

// Ray
template <> struct convert<PhysicsEngine::Ray>
{
    static Node encode(const PhysicsEngine::Ray &rhs)
    {
        Node node;
        node["origin"] = rhs.mOrigin;
        node["direction"] = rhs.mDirection;
        return node;
    }

    static bool decode(const Node &node, PhysicsEngine::Ray &rhs)
    {
        rhs.mOrigin = node["origin"].as<glm::vec3>();
        rhs.mDirection = node["direction"].as<glm::vec3>();

        return true;
    }
};

} // namespace YAML

#endif