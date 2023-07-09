#ifndef OPENGL_UNIFORM_BUFFER_H__
#define OPENGL_UNIFORM_BUFFER_H__

#include "../../UniformBuffer.h"

namespace PhysicsEngine
{
class OpenGLUniformBuffer : public UniformBuffer
{
  private:
    unsigned int mBuffer;
    unsigned int mBindingPoint;
    size_t mSize;

    char mData[2048];

  public:
    OpenGLUniformBuffer(size_t size, unsigned int bindingPoint);
    ~OpenGLUniformBuffer();

    size_t getSize() const override;
    unsigned int getBindingPoint() const override;

    void bind(PipelineStage stage) override;
    void unbind(PipelineStage stage) override;
    void setData(const void* data, size_t offset, size_t size) override;
    void getData(void* data, size_t offset, size_t size) override;
    void copyDataToDevice() override;
};
}

#endif