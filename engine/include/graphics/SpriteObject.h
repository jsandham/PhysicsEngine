#ifndef SPRITEOBJECT_H__
#define SPRITEOBJECT_H__

#define GLM_FORCE_RADIANS

#include "../glm/glm.hpp"
#include "../core/Color.h"

namespace PhysicsEngine
{
    typedef struct SpriteObject
    {
        glm::mat4 model;
        Color color;
        int vao;
        int texture;
    } SpriteObject;
} // namespace PhysicsEngine

#endif