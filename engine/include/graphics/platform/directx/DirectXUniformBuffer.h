#ifndef DIRECTX_UNIFORM_BUFFER_H__
#define DIRECTX_UNIFORM_BUFFER_H__

#include "../../UniformBuffer.h"

#include <windows.h>
#include <d3d11.h>

namespace PhysicsEngine
{
class DirectXUniformBuffer : public UniformBuffer
{
  private:
    unsigned int mBindingPoint;
    size_t mSize;

    D3D11_BUFFER_DESC mBufferDesc;
    D3D11_MAPPED_SUBRESOURCE mMappedSubresource;
    ID3D11Buffer *mBuffer;
    ID3D11InputLayout *mInputLayout;

  public:
    DirectXUniformBuffer(size_t size, unsigned int bindingPoint);
    ~DirectXUniformBuffer();

    size_t getSize() const override;
    unsigned int getBindingPoint() const override;

    void bind(PipelineStage stage) override;
    void unbind() override;
    void setData(const void *data, size_t offset, size_t size) override;
};
}

#endif