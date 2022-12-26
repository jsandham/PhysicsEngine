#include "../../include/graphics/RenderContext.h"
#include "../../include/graphics/MeshHandle.h"

#include "../../include/graphics/platform/directx/DirectXMeshHandle.h"
#include "../../include/graphics/platform/opengl/OpenGLMeshHandle.h"

using namespace PhysicsEngine;

MeshHandle::MeshHandle()
{
}

MeshHandle::~MeshHandle()
{
}

MeshHandle *MeshHandle::create()
{
    switch (RenderContext::getRenderAPI())
    {
    case RenderAPI::OpenGL:
        return new OpenGLMeshHandle();
    case RenderAPI::DirectX:
        return new DirectXMeshHandle();
    }

    return nullptr;
}