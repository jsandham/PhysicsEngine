#ifndef RENDEROBJECT_H__
#define RENDEROBJECT_H__

#define GLM_FORCE_RADIANS

#include "../core/Sphere.h"
#include "../glm/glm.hpp"

namespace PhysicsEngine
{
typedef struct RenderObject
{
    glm::mat4 model;
    Sphere boundingSphere;
    Guid transformId;
    Guid meshRendererId;
    int meshRendererIndex;
    int materialIndex;
    int shaderIndex;
    int start; // start index in vbo
    int size;  // size of vbo
    int vao;
    bool culled;
} RenderObject;
} // namespace PhysicsEngine

#endif