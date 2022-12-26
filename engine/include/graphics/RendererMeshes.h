#ifndef RENDERER_MESHES_H__
#define RENDERER_MESHES_H__

#include "ShaderProgram.h"
#include <string>

namespace PhysicsEngine
{
class FrustumMesh
{
  public:
    FrustumMesh()
    {
    }
    virtual ~FrustumMesh(){};

    virtual void bind() = 0;
    virtual void unbind() = 0;

    static FrustumMesh *create();
};

class GridMesh
{
  public:
    GridMesh()
    {
    }
    virtual ~GridMesh(){};

    virtual void bind() = 0;
    virtual void unbind() = 0;

    static GridMesh *create();
};


class RendererMeshes
{
  private:
    static FrustumMesh *sFrustumMesh;
    static GridMesh *sGridMesh;

  public:
    static FrustumMesh *getFrustumMesh();
    static GridMesh *getGridMesh();

    static void createInternalMeshes();
};
} // namespace PhysicsEngine

#endif