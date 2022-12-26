#ifndef OPENGL_RENDERER_MESHES_H__
#define OPENGL_RENDERER_MESHES_H__

#include "../../RendererMeshes.h"

namespace PhysicsEngine
{
class OpenGLFrustumMesh : public FrustumMesh
{
  public:
    OpenGLFrustumMesh();
    ~OpenGLFrustumMesh();

    void bind() override;
    void unbind() override;
};

class OpenGLGridMesh : public GridMesh
{
  public:
    OpenGLGridMesh();
    ~OpenGLGridMesh();

    void bind() override;
    void unbind() override;
};


} // namespace PhysicsEngine

#endif