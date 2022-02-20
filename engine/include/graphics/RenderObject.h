#ifndef RENDEROBJECT_H__
#define RENDEROBJECT_H__

#define GLM_FORCE_RADIANS

#include "../core/Sphere.h"
#include "glm/glm.hpp"

namespace PhysicsEngine
{
typedef struct RenderObject
{
    int instanceStart;
    int instanceCount;
    int materialIndex;
    int shaderIndex;
    int start; // start index in vbo
    int size;  // size of vbo
    int vao;
    int instanceModelVbo;
    int instanceColorVbo;
    bool instanced;

    bool operator==(const RenderObject &object) const
    {
        return this->instanceStart == object.instanceStart && this->instanceCount == object.instanceCount &&
               this->materialIndex == object.materialIndex && this->shaderIndex == object.shaderIndex &&
               this->start == object.start && this->size == object.size && this->vao == object.vao &&
               this->instanceModelVbo == object.instanceModelVbo && this->instanceColorVbo == object.instanceColorVbo &&
               this->instanced == object.instanced;
    }
} RenderObject;

struct InstanceModelData
{
    std::vector<glm::mat4> models;
    std::vector<Guid> transformIds;
    std::vector<Sphere> boundingSpheres;
};

struct pair_hash
{
    template <class T1, class T2> std::size_t operator()(const std::pair<T1, T2> &pair) const
    {
        return std::hash<T1>()(pair.first) ^ std::hash<T2>()(pair.second);
    }
};

typedef std::unordered_map<std::pair<Guid, RenderObject>, InstanceModelData, pair_hash> InstanceMap;
} // namespace PhysicsEngine

// allow use of InstancedRenderObject in unordered_set and unordered_map
namespace std
{
template <> struct hash<PhysicsEngine::RenderObject>
{
    size_t operator()(const PhysicsEngine::RenderObject &object) const noexcept
    {
        std::hash<int> hash;
        return hash(object.start) ^ hash(object.size) ^ hash(object.vao);
    }
};
} // namespace std

#endif