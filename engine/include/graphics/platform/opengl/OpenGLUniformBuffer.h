#ifndef OPENGL_UNIFORM_BUFFER_H__
#define OPENGL_UNIFORM_BUFFER_H__

#include "../../UniformBuffer.h"

namespace PhysicsEngine
{
class OpenGLUniformBuffer : public UniformBuffer
{
  private:
    unsigned int mBuffer;

  public:
    OpenGLUniformBuffer(size_t size);
    ~OpenGLUniformBuffer();

    void setData(void* data, size_t size, size_t offset, size_t bindingPoint) override;
};
}

#endif