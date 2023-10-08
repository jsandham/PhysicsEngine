#include "../../include/graphics/RenderObject.h"

#include <assert.h>

using namespace PhysicsEngine;

uint64_t PhysicsEngine::generateKey(int materialIndex, int meshIndex, int shaderIndex, int subMesh, int flags)
{
    assert(materialIndex >= std::numeric_limits<uint16_t>::min());
    assert(materialIndex <= std::numeric_limits<uint16_t>::max());
    assert(meshIndex >= std::numeric_limits<uint16_t>::min());
    assert(meshIndex <= std::numeric_limits<uint16_t>::max());
    assert(shaderIndex >= std::numeric_limits<uint16_t>::min());
    assert(shaderIndex <= std::numeric_limits<uint16_t>::max());
    assert(subMesh >= std::numeric_limits<uint8_t>::min());
    assert(subMesh <= std::numeric_limits<uint8_t>::max());
    assert(flags >= std::numeric_limits<uint8_t>::min());
    assert(flags <= std::numeric_limits<uint8_t>::max());

    uint64_t key = static_cast<uint64_t>(materialIndex);
    key += (static_cast<uint64_t>(meshIndex) << 16);
    key += (static_cast<uint64_t>(shaderIndex) << 32);
    key += (static_cast<uint64_t>(subMesh) << 48);
    key += (static_cast<uint64_t>(flags) << 56);

    return key;
}

uint16_t PhysicsEngine::getMaterialIndexFromKey(uint64_t key)
{
    uint64_t mask;
    mask = (((uint64_t)1 << 16) - (uint64_t)1) << 0;

    return static_cast<uint16_t>(key & mask);
}

uint16_t PhysicsEngine::getMeshIndexFromKey(uint64_t key)
{
    uint64_t mask;
    mask = (((uint64_t)1 << 16) - (uint64_t)1) << 16;
    return static_cast<uint16_t>((key & mask) >> 16);
}

uint16_t PhysicsEngine::getShaderIndexFromKey(uint64_t key)
{
    uint64_t mask;
    mask = (((uint64_t)1 << 16) - (uint64_t)1) << 32;
    return static_cast<uint16_t>((key & mask) >> 32);
}

uint8_t PhysicsEngine::getSubMeshFromKey(uint64_t key)
{
    uint64_t mask;
    mask = (((uint64_t)1 << 8) - (uint64_t)1) << 48;
    return static_cast<uint8_t>((key & mask) >> 48);
}

uint8_t PhysicsEngine::getFlagsFromKey(uint64_t key)
{
    uint64_t mask;
    mask = (((uint64_t)1 << 8) - (uint64_t)1) << 56;
    return static_cast<uint8_t>((key & mask) >> 56);
}

bool PhysicsEngine::isIndexed(uint64_t key)
{
    return static_cast<uint8_t>(getFlagsFromKey(key)) & static_cast<uint8_t>(RenderFlags::Indexed);
}

bool PhysicsEngine::isInstanced(uint64_t key)
{
    return static_cast<uint8_t>(getFlagsFromKey(key)) & static_cast<uint8_t>(RenderFlags::Instanced);
}