#ifndef DIRECTX_RENDERER_MESHES_H__
#define DIRECTX_RENDERER_MESHES_H__

#include "../../RendererMeshes.h"

namespace PhysicsEngine
{
class DirectXFrustumMesh : public FrustumMesh
{
  public:
    DirectXFrustumMesh();
    ~DirectXFrustumMesh();

    void bind() override;
    void unbind() override;
};

class DirectXGridMesh : public GridMesh
{
  public:
    DirectXGridMesh();
    ~DirectXGridMesh();

    void bind() override;
    void unbind() override;
};

} // namespace PhysicsEngine

#endif