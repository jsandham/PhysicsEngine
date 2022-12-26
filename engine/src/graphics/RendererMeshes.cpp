#include "../../include/core/Log.h"
#include "../../include/graphics/RenderContext.h"
#include "../../include/graphics/RendererMeshes.h"

#include "../../include/graphics/platform/directx/DirectXRendererMeshes.h"
#include "../../include/graphics/platform/opengl/OpenGLRendererMeshes.h"

using namespace PhysicsEngine;

FrustumMesh *RendererMeshes::sFrustumMesh = nullptr;
GridMesh *RendererMeshes::sGridMesh = nullptr;

void RendererMeshes::createInternalMeshes()
{
    // Note these pointers never free'd but they are static and
    // exist for the length of the program so ... meh?
    Log::warn("Start building internal meshes\n");
    sFrustumMesh = FrustumMesh::create();
    sGridMesh = GridMesh::create();
    Log::warn("Finished building internal meshes\n");
}

FrustumMesh *RendererMeshes::getFrustumMesh()
{
    return RendererMeshes::sFrustumMesh;
}

GridMesh *RendererMeshes::getGridMesh()
{
    return RendererMeshes::sGridMesh;
}

FrustumMesh *FrustumMesh::create()
{
    switch (RenderContext::getRenderAPI())
    {
    case RenderAPI::OpenGL:
        return new OpenGLFrustumMesh();
    case RenderAPI::DirectX:
        return new DirectXFrustumMesh();
    }

    return nullptr;
}

GridMesh *GridMesh::create()
{
    switch (RenderContext::getRenderAPI())
    {
    case RenderAPI::OpenGL:
        return new OpenGLGridMesh();
    case RenderAPI::DirectX:
        return new DirectXGridMesh();
    }

    return nullptr;
}

