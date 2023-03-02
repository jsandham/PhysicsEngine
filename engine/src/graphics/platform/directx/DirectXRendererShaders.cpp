#include "../../../../include/graphics/platform/directx/DirectXRendererShaders.h"

using namespace PhysicsEngine;

DirectXStandardShader::DirectXStandardShader()
{
}

DirectXStandardShader::~DirectXStandardShader()
{
}

void DirectXStandardShader::bind() 
{
}

void DirectXStandardShader::unbind()
{
}

std::string DirectXStandardShader::getVertexShader()
{
    return "";
}

std::string DirectXStandardShader::getFragmentShader()
{
    return "";
}

DirectXGBufferShader::DirectXGBufferShader()
{
}

DirectXGBufferShader::~DirectXGBufferShader()
{
}

void DirectXGBufferShader::bind()
{
}

void DirectXGBufferShader::unbind()
{
}

void DirectXGBufferShader::setModel(const glm::mat4 &model)
{
}

DirectXQuadShader::DirectXQuadShader()
{
}

DirectXQuadShader::~DirectXQuadShader()
{
}

void DirectXQuadShader::bind()
{
}

void DirectXQuadShader::unbind()
{
}

void DirectXQuadShader::setScreenTexture(int texUnit, TextureHandle *tex)
{
}

DirectXDepthShader::DirectXDepthShader()
{
}

DirectXDepthShader::~DirectXDepthShader()
{
}

void DirectXDepthShader::bind()
{
}

void DirectXDepthShader::unbind()
{
}

void DirectXDepthShader::setModel(const glm::mat4 &model)
{
}

void DirectXDepthShader::setView(const glm::mat4 &view)
{
}

void DirectXDepthShader::setProjection(const glm::mat4 &projection)
{
}

DirectXDepthCubemapShader::DirectXDepthCubemapShader()
{
}

DirectXDepthCubemapShader::~DirectXDepthCubemapShader()
{
}

void DirectXDepthCubemapShader::bind()
{
}

void DirectXDepthCubemapShader::unbind()
{
}

void DirectXDepthCubemapShader::setLightPos(const glm::vec3 &lightPos)
{
}

void DirectXDepthCubemapShader::setFarPlane(float farPlane)
{
}

void DirectXDepthCubemapShader::setModel(const glm::mat4 &model)
{
}

void DirectXDepthCubemapShader::setCubeViewProj(int index, const glm::mat4 &modelView)
{
}

DirectXGeometryShader::DirectXGeometryShader()
{
}

DirectXGeometryShader::~DirectXGeometryShader()
{
}

void DirectXGeometryShader::bind()
{
}

void DirectXGeometryShader::unbind()
{
}

void DirectXGeometryShader::setModel(const glm::mat4 &model)
{
}

DirectXNormalShader::DirectXNormalShader()
{
}

DirectXNormalShader::~DirectXNormalShader()
{
}

void DirectXNormalShader::bind()
{
}

void DirectXNormalShader::unbind()
{
}

void DirectXNormalShader::setModel(const glm::mat4 &model)
{
}

DirectXNormalInstancedShader::DirectXNormalInstancedShader()
{
}

DirectXNormalInstancedShader::~DirectXNormalInstancedShader()
{
}

void DirectXNormalInstancedShader::bind()
{
}

void DirectXNormalInstancedShader::unbind()
{
}

DirectXPositionShader::DirectXPositionShader()
{
}

DirectXPositionShader::~DirectXPositionShader()
{
}

void DirectXPositionShader::bind()
{
}

void DirectXPositionShader::unbind()
{
}

void DirectXPositionShader::setModel(const glm::mat4 &model)
{
}

DirectXPositionInstancedShader::DirectXPositionInstancedShader()
{
}

DirectXPositionInstancedShader::~DirectXPositionInstancedShader()
{
}

void DirectXPositionInstancedShader::bind()
{
}

void DirectXPositionInstancedShader::unbind()
{
}

DirectXLinearDepthShader::DirectXLinearDepthShader()
{   
}

DirectXLinearDepthShader::~DirectXLinearDepthShader()
{
}

void DirectXLinearDepthShader::bind()
{
}

void DirectXLinearDepthShader::unbind()
{
}

void DirectXLinearDepthShader::setModel(const glm::mat4 &model)
{
}

DirectXLinearDepthInstancedShader::DirectXLinearDepthInstancedShader()
{
}

DirectXLinearDepthInstancedShader::~DirectXLinearDepthInstancedShader()
{
}

void DirectXLinearDepthInstancedShader::bind()
{
}

void DirectXLinearDepthInstancedShader::unbind()
{
}

DirectXColorShader::DirectXColorShader()
{
}

DirectXColorShader::~DirectXColorShader()
{
}

void DirectXColorShader::bind()
{
}

void DirectXColorShader::unbind()
{
}

void DirectXColorShader::setModel(const glm::mat4 &model)
{
}

void DirectXColorShader::setColor32(const Color32 &color)
{
}

DirectXColorInstancedShader::DirectXColorInstancedShader()
{
}

DirectXColorInstancedShader::~DirectXColorInstancedShader()
{
}

void DirectXColorInstancedShader::bind()
{
}

void DirectXColorInstancedShader::unbind()
{
}

DirectXSSAOShader::DirectXSSAOShader()
{
}

DirectXSSAOShader::~DirectXSSAOShader()
{
}

void DirectXSSAOShader::bind()
{
}

void DirectXSSAOShader::unbind()
{
}

void DirectXSSAOShader::setProjection(const glm::mat4 &projection)
{
}

void DirectXSSAOShader::setPositionTexture(int texUnit, TextureHandle *tex)
{
}

void DirectXSSAOShader::setNormalTexture(int texUnit, TextureHandle *tex)
{
}

void DirectXSSAOShader::setNoiseTexture(int texUnit, TextureHandle *tex)
{
}

void DirectXSSAOShader::setSample(int index, const glm::vec3 &sample)
{
}

DirectXSpriteShader::DirectXSpriteShader()
{
}

DirectXSpriteShader::~DirectXSpriteShader()
{
}

void DirectXSpriteShader::bind()
{
}

void DirectXSpriteShader::unbind()
{
}

void DirectXSpriteShader::setModel(const glm::mat4 &model)
{
}

void DirectXSpriteShader::setView(const glm::mat4 &view)
{
}

void DirectXSpriteShader::setProjection(const glm::mat4 &projection)
{
}

void DirectXSpriteShader::setColor(const Color &color)
{
}

void DirectXSpriteShader::setImage(int texUnit, TextureHandle *tex)
{
}

DirectXLineShader::DirectXLineShader()
{
}

DirectXLineShader::~DirectXLineShader()
{
}

void DirectXLineShader::bind()
{
}

void DirectXLineShader::unbind()
{
}

void DirectXLineShader::setMVP(const glm::mat4 &mvp)
{
}

DirectXGizmoShader::DirectXGizmoShader()
{
}

DirectXGizmoShader::~DirectXGizmoShader()
{
}

void DirectXGizmoShader::bind()
{
}

void DirectXGizmoShader::unbind()
{
}

void DirectXGizmoShader::setModel(const glm::mat4 &model)
{
}

void DirectXGizmoShader::setView(const glm::mat4 &view)
{
}

void DirectXGizmoShader::setProjection(const glm::mat4 &projection)
{
}

void DirectXGizmoShader::setColor(const Color &color)
{
}

void DirectXGizmoShader::setLightPos(const glm::vec3 &lightPos)
{
}

DirectXGridShader::DirectXGridShader()
{
}

DirectXGridShader::~DirectXGridShader()
{
}

void DirectXGridShader::bind()
{
}

void DirectXGridShader::unbind()
{
}

void DirectXGridShader::setMVP(const glm::mat4 &mvp)
{
}

void DirectXGridShader::setColor(const Color &color)
{
}