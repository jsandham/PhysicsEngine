#ifndef DIRECTX_MESHHANDLE_H__
#define DIRECTX_MESHHANDLE_H__

#include <vector>

#include "../../MeshHandle.h"
#include "../../VertexBuffer.h"

namespace PhysicsEngine
{
class DirectXMeshHandle : public MeshHandle
{
  public:
    DirectXMeshHandle();
    ~DirectXMeshHandle();

    void bind() override;
    void unbind() override;
    void setData(void *data, size_t offset, size_t size, MeshVBO meshVBO) override;
    void draw() override;
    VertexBuffer *getVBO(MeshVBO meshVBO) override;
    unsigned int getVAO() override;
};
} // namespace PhysicsEngine

#endif