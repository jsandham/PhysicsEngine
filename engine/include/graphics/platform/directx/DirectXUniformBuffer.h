#ifndef DIRECTX_UNIFORM_BUFFER_H__
#define DIRECTX_UNIFORM_BUFFER_H__

#include "../../UniformBuffer.h"

namespace PhysicsEngine
{
class DirectXUniformBuffer : public UniformBuffer
{
  public:
    DirectXUniformBuffer(size_t size);
    ~DirectXUniformBuffer();

    void setData(void *data, size_t size, size_t offset, size_t bindingPoint) override;
};
}

#endif