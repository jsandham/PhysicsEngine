#include "../../include/graphics/RenderObject.h"

#include <assert.h>

using namespace PhysicsEngine;

DrawCallCommand::DrawCallCommand() : mDrawCallCode(0)
{
}

DrawCallCommand::DrawCallCommand(uint64_t drawCallCode) : mDrawCallCode(drawCallCode)
{
}

uint64_t DrawCallCommand::getCode() const
{
    return mDrawCallCode;
}

void DrawCallCommand::generateTerrainDrawCall(int materialIndex, int terrainIndex, int chunk, int flags)
{
    mDrawCallCode = 0;

    // [reserved ][  chunk  ][terrain index][ material index]
    // [ 16 bits ][ 16 bits ][   16 bits   ][    16-bits    ]
    // 64                                                   0
    assert((materialIndex >= 0 && materialIndex <= 65535));
    assert((terrainIndex >= 0 && terrainIndex <= 65535));
    assert((chunk >= 0 && chunk <= 65535));
    assert((flags >= 0 && flags <= 65535));

    mDrawCallCode = static_cast<uint64_t>(materialIndex);
    mDrawCallCode += (static_cast<uint64_t>(terrainIndex) << 16);
    mDrawCallCode += (static_cast<uint64_t>(chunk) << 32);
    mDrawCallCode += (static_cast<uint64_t>(flags) << 48);
}

void DrawCallCommand::generateDrawCall(int materialIndex, int meshIndex, int subMesh, int depth, int flags)
{
    mDrawCallCode = 0;

    // [reserved][  depth  ][sub mesh][mesh index][ material index]
    // [ 5 bits ][ 24-bits ][ 3 bits ][ 16 bits  ][    16-bits    ]
    // 64                                                         0
    assert((materialIndex >= 0 && materialIndex <= 65535));
    assert((meshIndex >= 0 && meshIndex <= 65535));
    assert((subMesh >= 0 && subMesh <= 7));
    assert((depth >= 0 && depth <= 16777215));
    assert((flags >= 0 && flags <= 31));

    mDrawCallCode = static_cast<uint64_t>(materialIndex);
    mDrawCallCode += (static_cast<uint64_t>(meshIndex) << 16);
    mDrawCallCode += (static_cast<uint64_t>(subMesh) << 32);
    mDrawCallCode += (static_cast<uint64_t>(depth) << 35);
    mDrawCallCode += (static_cast<uint64_t>(flags) << 59);
}

uint16_t DrawCallCommand::getMaterialIndex() const
{
    uint64_t mask;
    mask = (((uint64_t)1 << 16) - (uint64_t)1) << 0;

    return static_cast<uint16_t>(mDrawCallCode & mask);
}

uint16_t DrawCallCommand::getMeshIndex() const
{
    uint64_t mask;
    mask = (((uint64_t)1 << 16) - (uint64_t)1) << 16;

    return static_cast<uint16_t>((mDrawCallCode & mask) >> 16);
}

uint8_t DrawCallCommand::getSubMesh() const
{
    uint64_t mask;
    mask = (((uint64_t)1 << 3) - (uint64_t)1) << 32;

    return static_cast<uint8_t>((mDrawCallCode & mask) >> 32);
}

uint32_t DrawCallCommand::getDepth() const
{
    uint64_t mask;
    mask = (((uint64_t)1 << 24) - (uint64_t)1) << 35;

    return static_cast<uint32_t>((mDrawCallCode & mask) >> 35);
}

uint8_t DrawCallCommand::getFlags() const
{
    uint64_t mask;
    mask = (((uint64_t)1 << 5) - (uint64_t)1) << 59;

    return static_cast<uint32_t>((mDrawCallCode & mask) >> 59);
}

void DrawCallCommand::markDrawCallAsInstanced()
{
    if (!isInstanced())
    {
        mDrawCallCode += (static_cast<uint64_t>(DrawCallFlags::Instanced) << 59);
    }
}

void DrawCallCommand::markDrawCallAsIndexed()
{
    if (!isIndexed())
    {
        mDrawCallCode += (static_cast<uint64_t>(DrawCallFlags::Indexed) << 59);
    }
}

void DrawCallCommand::markDrawCallAsTerrain()
{
    if (!isTerrain())
    {
        mDrawCallCode += (static_cast<uint64_t>(DrawCallFlags::Terrain) << 48);
    }
}

bool DrawCallCommand::isIndexed() const
{
    return static_cast<uint8_t>(getFlags()) & static_cast<uint8_t>(DrawCallFlags::Indexed);
}

bool DrawCallCommand::isInstanced() const
{
    return static_cast<uint8_t>(getFlags()) & static_cast<uint8_t>(DrawCallFlags::Instanced);
}

bool DrawCallCommand::isTerrain() const
{
    return static_cast<uint8_t>(getFlags()) & static_cast<uint8_t>(DrawCallFlags::Terrain);
}