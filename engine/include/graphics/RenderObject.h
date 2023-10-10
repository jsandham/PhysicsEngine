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
    int instanceCount;
    uint64_t key;
} RenderObject;

enum class RenderFlags
{
    Indexed = 1,
    Instanced = 2
};

uint64_t generateDrawCall(int materialIndex, int meshIndex, int shaderIndex, int subMesh, int flags);
uint16_t getMaterialIndexFromKey(uint64_t key);
uint16_t getMeshIndexFromKey(uint64_t key);
uint16_t getShaderIndexFromKey(uint64_t key);
uint8_t getSubMeshFromKey(uint64_t key);
uint8_t getFlagsFromKey(uint64_t key);
bool isIndexed(uint64_t key);
bool isInstanced(uint64_t key);
} // namespace PhysicsEngine
#endif