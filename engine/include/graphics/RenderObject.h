#ifndef RENDEROBJECT_H__
#define RENDEROBJECT_H__

#include "../core/Material.h"

#include "../graphics/MeshHandle.h"

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
    int meshRendererIndex;
    bool indexed;
} DrawCallCommand;

uint64_t generateDrawCall(int materialIndex, int meshIndex, int subMesh, int depth);
uint16_t getMaterialIndexFromKey(uint64_t key);
uint16_t getMeshIndexFromKey(uint64_t key);
uint8_t getSubMeshFromKey(uint64_t key);
uint32_t getDepthFromKey(uint64_t key);
} // namespace PhysicsEngine
#endif