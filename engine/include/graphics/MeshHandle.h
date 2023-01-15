#ifndef MESHHANDLE_H__
#define MESHHANDLE_H__

#include "VertexBuffer.h"

namespace PhysicsEngine
{
enum class MeshVBO
{
    Vertices,
    Normals,
    TexCoords,
    InstanceModel,
    InstanceColor
};

class MeshHandle
{
  public:
    MeshHandle();
    MeshHandle(const MeshHandle &other) = delete;
    MeshHandle &operator=(const MeshHandle &other) = delete;
    virtual ~MeshHandle() = 0;

    virtual void bind() = 0;
    virtual void unbind() = 0;
    virtual void setData(void* data, size_t offset, size_t size, MeshVBO meshVBO) = 0;
    virtual void draw() = 0;
    virtual VertexBuffer *getVBO(MeshVBO meshVBO) = 0;
    virtual unsigned int getVAO() = 0;

    static MeshHandle* create();
};
}

#endif