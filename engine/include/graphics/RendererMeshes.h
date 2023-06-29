#ifndef RENDERER_MESHES_H__
#define RENDERER_MESHES_H__

#include "MeshHandle.h"

namespace PhysicsEngine
{

class ScreenQuad
{
  private:
    MeshHandle *mMesh;

    VertexBuffer *mVertexBuffer;
    VertexBuffer *mTexCoordsBuffer;

  public:
    ScreenQuad();
    ~ScreenQuad();

    void bind();
    void unbind();
    void draw();
};

class RendererMeshes
{
  private:
    static ScreenQuad *sScreenQuad;

  public:
    static ScreenQuad *getScreenQuad();
   
    static void createInternalMeshes();
};
} // namespace PhysicsEngine

#endif