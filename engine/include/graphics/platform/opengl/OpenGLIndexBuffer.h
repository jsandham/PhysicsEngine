#ifndef OPENGL_INDEX_BUFFER_H__
#define OPENGL_INDEX_BUFFER_H__

#include "../../IndexBuffer.h"

namespace PhysicsEngine
{
class OpenGLIndexBuffer : public IndexBuffer
{
  public:
    unsigned int mBuffer;

  public:
    OpenGLIndexBuffer();
    ~OpenGLIndexBuffer();

    void resize(size_t size) override;
    void setData(const void *data, size_t offset, size_t size) override;
    void bind() override;
    void unbind() override;
    void *getBuffer() override;
};
} // namespace PhysicsEngine

#endif