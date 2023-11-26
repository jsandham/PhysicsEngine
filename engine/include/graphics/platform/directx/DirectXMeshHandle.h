#ifndef DIRECTX_MESHHANDLE_H__
#define DIRECTX_MESHHANDLE_H__

#include <vector>

#include "../../MeshHandle.h"

#include <d3d11.h>

namespace PhysicsEngine
{
class DirectXMeshHandle : public MeshHandle
{
  private:
    std::vector<VertexBuffer *> mBuffers;
    IndexBuffer *mIndexBuffer;

    ID3D11InputLayout *mBufferLayout;
    std::vector<std::string> mInputSemanticNames;
    std::vector<D3D11_INPUT_ELEMENT_DESC> mInputDescs; 

  public:
    DirectXMeshHandle();
    ~DirectXMeshHandle();

    void addVertexBuffer(VertexBuffer *buffer, std::string name, AttribType type, bool instanceBuffer = false) override;
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