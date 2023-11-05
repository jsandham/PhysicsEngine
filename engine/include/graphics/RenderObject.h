#ifndef RENDEROBJECT_H__
#define RENDEROBJECT_H__

#define GLM_FORCE_RADIANS

#include "../core/Guid.h"
#include "../core/Id.h"
#include "../core/Sphere.h"
#include "../core/Material.h"

#include "../graphics/MeshHandle.h"

#include "glm/glm.hpp"

namespace PhysicsEngine
{
typedef struct DrawCallCommand
{
    MeshHandle *meshHandle;
    VertexBuffer *instanceModelBuffer;
    VertexBuffer *instanceColorBuffer;
    Material *material;
    Shader *shader;
    int meshStartIndex;
    int meshEndIndex;
    int instanceCount;
    bool indexed;
} DrawCallCommand;

enum class RenderFlags
{
    Indexed = 1,
    Instanced = 2,
    Terrain = 4
};

uint64_t generateDrawCall(int materialIndex, int meshIndex, int shaderIndex, int subMesh, int flags);
uint16_t getMaterialIndexFromKey(uint64_t key);
uint16_t getMeshIndexFromKey(uint64_t key);
uint16_t getShaderIndexFromKey(uint64_t key);
uint8_t getSubMeshFromKey(uint64_t key);
uint8_t getFlagsFromKey(uint64_t key);
bool isIndexed(uint64_t key);
bool isInstanced(uint64_t key);
bool isTerrain(uint64_t key);
} // namespace PhysicsEngine
#endif