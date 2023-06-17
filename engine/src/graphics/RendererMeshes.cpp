#include "../../include/core/Log.h"

#include "../../include/graphics/RendererMeshes.h"

using namespace PhysicsEngine;

// Meshes generated using clockwise triangle winding order. Note: For OpenGL, the default 
// windng order is counter-clockwise (CCW) while for DirectX the default is clockwise (CW)


SphereMesh::SphereMesh()
{
    mMesh = MeshHandle::create();

}

SphereMesh::~SphereMesh()
{
    delete mMesh;
}

void SphereMesh::bind()
{
    mMesh->bind();
}

void SphereMesh::unbind()
{
    mMesh->unbind();
}

SphereMesh *RendererMeshes::sSphereMesh = nullptr;

void RendererMeshes::createInternalMeshes()
{
    // Note these pointers never free'd but they are static and
    // exist for the length of the program so ... meh?
    Log::warn("Start building internal meshes\n");
    sSphereMesh = new SphereMesh();
    Log::warn("Finished building internal meshes\n");
}

SphereMesh *RendererMeshes::getSphereMesh()
{
    return RendererMeshes::sSphereMesh;
}