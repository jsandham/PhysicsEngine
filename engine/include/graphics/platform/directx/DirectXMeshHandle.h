#ifndef DIRECTX_MESHHANDLE_H__
#define DIRECTX_MESHHANDLE_H__

#include <vector>

#include "../../MeshHandle.h"
#include "../../VertexBuffer.h"

namespace PhysicsEngine
{
class DirectXMeshHandle : public MeshHandle
{
  public:
    DirectXMeshHandle();
    ~DirectXMeshHandle();

    void addVertexBuffer(VertexBuffer *buffer, AttribType type) override;
    void bind() override;
    void unbind() override;
    void drawLines(size_t vertexOffset, size_t vertexCount) override;
    void draw(size_t vertexOffset, size_t vertexCount) override;
    void drawInstanced(size_t vertexOffset, size_t vertexCount, size_t instanceCount) override;
};
} // namespace PhysicsEngine

#endif