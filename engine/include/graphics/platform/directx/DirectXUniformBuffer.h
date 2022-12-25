#ifndef DIRECTX_UNIFORM_BUFFER_H__
#define DIRECTX_UNIFORM_BUFFER_H__

#include "../../UniformBuffer.h"

namespace PhysicsEngine
{
class DirectXUniformBuffer : public UniformBuffer
{
  private:
    unsigned int mBindingPoint;
    size_t mSize;

  public:
    DirectXUniformBuffer(size_t size, unsigned int bindingPoint);
    ~DirectXUniformBuffer();

    size_t getSize() const override;
    unsigned int getBindingPoint() const override;

    void bind() override;
    void unbind() override;
    void setData(void *data, size_t offset, size_t size) override;
};
}

#endif