#ifndef MESHHANDLE_H__
#define MESHHANDLE_H__

#include "VertexBuffer.h"

namespace PhysicsEngine
{
enum class AttribType
{
    Vec2,
    Vec3,
    Vec4,
    Mat4
};

class MeshHandle
{
  public:
    MeshHandle();
    MeshHandle(const MeshHandle &other) = delete;
    MeshHandle &operator=(const MeshHandle &other) = delete;
    virtual ~MeshHandle() = 0;

    virtual void addVertexBuffer(VertexBuffer* buffer, AttribType type) = 0;
    virtual void bind() = 0;
    virtual void unbind() = 0;
    virtual void drawLines(size_t vertexOffset, size_t vertexCount) = 0;
    virtual void draw(size_t vertexOffset, size_t vertexCount) = 0;
    virtual void drawInstanced(size_t vertexOffset, size_t vertexCount, size_t instanceCount) = 0;
    
    static MeshHandle* create();
};
}

#endif