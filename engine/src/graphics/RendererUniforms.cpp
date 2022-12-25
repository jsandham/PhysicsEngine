#include "../../include/graphics/RendererUniforms.h"
#include "../../include/graphics/RenderContext.h"
#include "../../include/graphics/RendererUniforms.h"
#include "../../include/core/Log.h"

#include "../../include/graphics/platform/opengl/OpenGLRendererUniforms.h"
#include "../../include/graphics/platform/directx/DirectXRendererUniforms.h"

using namespace PhysicsEngine;

CameraUniform* RendererUniforms::sCameraUniform = nullptr;
LightUniform* RendererUniforms::sLightUniform = nullptr;

void RendererUniforms::createInternalUniforms()
{
    // Note these pointers never free'd but they are static and 
    // exist for the length of the program so ... meh?
    sCameraUniform = CameraUniform::create();
    sLightUniform = LightUniform::create();
}

CameraUniform* RendererUniforms::getCameraUniform()
{
    return RendererUniforms::sCameraUniform;
}

LightUniform* RendererUniforms::getLightUniform()
{
    return RendererUniforms::sLightUniform;
}

CameraUniform* CameraUniform::create()
{
    switch (RenderContext::getRenderAPI())
    {
    case RenderAPI::OpenGL:
        return new OpenGLCameraUniform();
    case RenderAPI::DirectX:
        return new DirectXCameraUniform();
    }

    return nullptr;
}

LightUniform *LightUniform::create()
{
    switch (RenderContext::getRenderAPI())
    {
    case RenderAPI::OpenGL:
        return new OpenGLLightUniform();
    case RenderAPI::DirectX:
        return new DirectXLightUniform();
    }

    return nullptr;
}

