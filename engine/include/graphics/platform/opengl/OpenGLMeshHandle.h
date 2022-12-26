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
    VertexBuffer *mVbo[5];

  public:
    OpenGLMeshHandle();
    ~OpenGLMeshHandle();

    void bind() override;
    void unbind() override;
    void setData(void* data, size_t offset, size_t size, MeshVBO meshVBO) override;
    void draw() override;
    VertexBuffer *getVBO(MeshVBO meshVBO) override;
    unsigned int getVAO() override;
};
} // namespace PhysicsEngine

#endif