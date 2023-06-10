#ifndef RENDERER_MESHES_H__
#define RENDERER_MESHES_H__

#include "MeshHandle.h"

namespace PhysicsEngine
{
class PlaneMesh
{
  private:
    MeshHandle *mMesh;

  public:
    PlaneMesh(int nx, int nz);
    ~PlaneMesh();

    void bind();
    void unbind();
};

class SphereMesh
{
  private:
    MeshHandle *mMesh;

  public:
    SphereMesh();
    ~SphereMesh();

    void bind();
    void unbind();
};

class RendererMeshes
{
  private:
    static SphereMesh *sSphereMesh;

  public:
    static SphereMesh *getSphereMesh();
   
    static void createInternalMeshes();
};
} // namespace PhysicsEngine

#endif