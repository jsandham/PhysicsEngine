#ifndef DIRECTX_MESHHANDLE_H__
#define DIRECTX_MESHHANDLE_H__

#include <vector>

#include "../../MeshHandle.h"

namespace PhysicsEngine
{
class DirectXMeshHandle : public MeshHandle
{
  public:
    DirectXMeshHandle();
    ~DirectXMeshHandle();

    void addVertexBuffer(VertexBuffer *buffer, AttribType type, bool instanceBuffer = false) override;
    void addIndexBuffer(IndexBuffer *buffer) override;
    void bind() override;
    void unbind() override;
    void drawLines(size_t vertexOffset, size_t vertexCount) override;
    void draw(size_t vertexOffset, size_t vertexCount) override;
    void drawIndexed(size_t vertexOffset, size_t vertexCount) override;
    void drawInstanced(size_t vertexOffset, size_t vertexCount, size_t instanceCount) override;
    void drawIndexedInstanced(size_t vertexOffset, size_t vertexCount, size_t instanceCount) override;
};
} // namespace PhysicsEngine

#endif