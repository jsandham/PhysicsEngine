#include "../../include/graphics/ShaderProgram.h"
#include "../../include/graphics/RenderContext.h"

#include "../../include/graphics/platform/directx/DirectXShaderProgram.h"
#include "../../include/graphics/platform/opengl/OpenGLShaderProgram.h"

using namespace PhysicsEngine;

ShaderProgram::ShaderProgram()
{
}

ShaderProgram::~ShaderProgram()
{
}

std::string ShaderProgram::getVertexShader() const
{
    return mVertex;
}

std::string ShaderProgram::getFragmentShader() const
{
    return mFragment;
}

std::string ShaderProgram::getGeometryShader() const
{
    return mGeometry;
}

ShaderStatus ShaderProgram::getStatus() const
{
    return mStatus;
}

ShaderProgram *ShaderProgram::create()
{
    switch (RenderContext::getRenderAPI())
    {
    case RenderAPI::OpenGL:
        return new OpenGLShaderProgram();
    case RenderAPI::DirectX:
        return new DirectXShaderProgram();
    }

    return nullptr;
}