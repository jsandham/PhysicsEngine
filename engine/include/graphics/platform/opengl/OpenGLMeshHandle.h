#ifndef OPENGL_MESHHANDLE_H__
#define OPENGL_MESHHANDLE_H__

#include <vector>
#include <glm/glm.hpp>

#include "../../MeshHandle.h"
#include "../../VertexBuffer.h"

namespace PhysicsEngine
{
class OpenGLMeshHandle : public MeshHandle
{
  private:
    unsigned int mVao;
    unsigned int mVertexAttribIndex;
    std::vector<VertexBuffer *> mBuffers;

  public:
    OpenGLMeshHandle();
    ~OpenGLMeshHandle();

    void bind() override;
    void unbind() override;
    void addVertexBuffer(VertexBuffer* buffer, AttribType type) override;
    void drawLines(size_t vertexOffset, size_t vertexCount) override;
    void draw(size_t vertexOffset, size_t vertexCount) override;
    void drawInstanced(size_t vertexOffset, size_t vertexCount, size_t instanceCount) override;
};
} // namespace PhysicsEngine

#endif