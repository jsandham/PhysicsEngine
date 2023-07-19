#ifndef DIRECTX_INDEX_BUFFER_H__
#define DIRECTX_INDEX_BUFFER_H__

#include "../../IndexBuffer.h"

#include <d3d11.h>
#include <windows.h>

namespace PhysicsEngine
{
class DirectXIndexBuffer : public IndexBuffer
{
  private:
    D3D11_BUFFER_DESC mBufferDesc;
    ID3D11Buffer *mBuffer;

  public:
    DirectXIndexBuffer();
    ~DirectXIndexBuffer();

    void resize(size_t size) override;
    void setData(void *data, size_t offset, size_t size) override;
    void bind() override;
    void unbind() override;
    void *getBuffer() override;
};
} // namespace PhysicsEngine

#endif