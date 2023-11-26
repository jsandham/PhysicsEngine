#ifndef OPENGL_MESHHANDLE_H__
#define OPENGL_MESHHANDLE_H__

#include <vector>
#include <string>

#include "../../MeshHandle.h"

namespace PhysicsEngine
{
class OpenGLMeshHandle : public MeshHandle
{
  private:
    unsigned int mVao;
    unsigned int mVertexAttribIndex;
    std::vector<VertexBuffer *> mBuffers;
    IndexBuffer *mIndexBuffer;

  public:
    OpenGLMeshHandle();
    ~OpenGLMeshHandle();

    void bind() override;
    void unbind() override;
    void addVertexBuffer(VertexBuffer *buffer, std::string name, AttribType type, bool instanceBuffer = false) override;
    void addIndexBuffer(IndexBuffer *buffer) override;
    void drawLines(size_t vertexOffset, size_t vertexCount) override;
    void draw(size_t vertexOffset, size_t vertexCount) override;
    void drawIndexed(size_t indexOffset, size_t indexCount) override;
    void drawInstanced(size_t vertexOffset, size_t vertexCount, size_t instanceCount) override;
    void drawIndexedInstanced(size_t indexOffset, size_t indexCount, size_t instanceCount) override;
};
} // namespace PhysicsEngine

#endif