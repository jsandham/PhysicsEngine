#ifndef DIRECTX_VERTEX_BUFFER_H__
#define DIRECTX_VERTEX_BUFFER_H__

#include "../../VertexBuffer.h"

#include <d3d11.h>
#include <windows.h>

namespace PhysicsEngine
{
class DirectXVertexBuffer : public VertexBuffer
{
  private:
    D3D11_BUFFER_DESC mBufferDesc;
    D3D11_MAPPED_SUBRESOURCE mMappedSubresource;
    ID3D11Buffer *mBuffer;
    ID3D11InputLayout *mInputLayout;

  public:
    DirectXVertexBuffer();
    ~DirectXVertexBuffer();

    void resize(size_t size) override;
    void setData(const void *data, size_t offset, size_t size) override;
    void bind() override;
    void unbind() override;
    void *getBuffer() override;
};
} // namespace PhysicsEngine

#endif