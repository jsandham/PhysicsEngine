#ifndef SPRITEOBJECT_H__
#define SPRITEOBJECT_H__

#define GLM_FORCE_RADIANS

#include "glm/glm.hpp"
#include "../core/Color.h"

#include "MeshHandle.h"
#include "TextureHandle.h"

namespace PhysicsEngine
{
    typedef struct SpriteObject
    {
        glm::mat4 model;
        Color color;
        MeshHandle *mHandle;
        TextureHandle *mTexture;
    } SpriteObject;
} // namespace PhysicsEngine

#endif