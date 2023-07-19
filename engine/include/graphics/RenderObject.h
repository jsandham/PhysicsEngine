#ifndef RENDEROBJECT_H__
#define RENDEROBJECT_H__

#define GLM_FORCE_RADIANS

#include "../core/Guid.h"
#include "../core/Id.h"
#include "../core/Sphere.h"
#include "../graphics/MeshHandle.h"

#include "glm/glm.hpp"

namespace PhysicsEngine
{
typedef struct RenderObject
{
    MeshHandle *meshHandle;
    VertexBuffer *instanceModelBuffer;
    VertexBuffer *instanceColorBuffer;

    int instanceStart;
    int instanceCount;
    int materialIndex;
    int shaderIndex;
    int start; // start index in vbo
    int size;  // size of vbo
    bool instanced;
    bool indexed;

    bool operator==(const RenderObject &object) const
    {
        return this->instanceStart == object.instanceStart && this->instanceCount == object.instanceCount &&
               this->materialIndex == object.materialIndex && this->shaderIndex == object.shaderIndex &&
               this->start == object.start && this->size == object.size && this->instanced == object.instanced &&
               this->indexed == object.indexed && this->meshHandle == object.meshHandle &&
               this->instanceModelBuffer == object.instanceModelBuffer &&
               this->instanceColorBuffer == object.instanceColorBuffer;
    }
} RenderObject;

struct InstanceModelData
{
    std::vector<glm::mat4> models;
    std::vector<Id> transformIds;
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
        return hash(object.start) ^ hash(object.size);
    }
};
} // namespace std

#endif