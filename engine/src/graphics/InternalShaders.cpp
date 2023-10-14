#include "../../include/graphics/InternalShaders.h"

#include "../../include/graphics/RenderContext.h"

#include "platform/directx/HLSL/hlsl_shaders.h"
#include "platform/opengl/GLSL/glsl_shaders.h"

using namespace PhysicsEngine;

#define INSTANIATE_SHADER(name)                                                                                        \
    std::string PhysicsEngine::name()                                                                                  \
    {                                                                                                                  \
        switch (RenderContext::getRenderAPI())                                                                         \
        {                                                                                                              \
        case RenderAPI::OpenGL:                                                                                        \
            return glsl::name();                                                                                       \
        case RenderAPI::DirectX:                                                                                       \
            return hlsl::name();                                                                                       \
        default:                                                                                                       \
            return std::string();                                                                                      \
        }                                                                                                              \
    }

INSTANIATE_SHADER(getGeometryVertexShader)
INSTANIATE_SHADER(getGeometryFragmentShader)
INSTANIATE_SHADER(getSSAOVertexShader)
INSTANIATE_SHADER(getSSAOFragmentShader)
INSTANIATE_SHADER(getShadowDepthMapVertexShader)
INSTANIATE_SHADER(getShadowDepthMapFragmentShader)
INSTANIATE_SHADER(getShadowDepthCubemapVertexShader)
INSTANIATE_SHADER(getShadowDepthCubemapFragmentShader)
INSTANIATE_SHADER(getShadowDepthCubemapGeometryShader)
INSTANIATE_SHADER(getColorVertexShader)
INSTANIATE_SHADER(getColorFragmentShader)
INSTANIATE_SHADER(getColorInstancedVertexShader)
INSTANIATE_SHADER(getColorInstancedFragmentShader)
INSTANIATE_SHADER(getScreenQuadVertexShader)
INSTANIATE_SHADER(getScreenQuadFragmentShader)
INSTANIATE_SHADER(getSpriteVertexShader)
INSTANIATE_SHADER(getSpriteFragmentShader)
INSTANIATE_SHADER(getGBufferVertexShader)
INSTANIATE_SHADER(getGBufferFragmentShader)
INSTANIATE_SHADER(getNormalVertexShader)
INSTANIATE_SHADER(getNormalFragmentShader)
INSTANIATE_SHADER(getNormalInstancedVertexShader)
INSTANIATE_SHADER(getNormalInstancedFragmentShader)
INSTANIATE_SHADER(getPositionVertexShader)
INSTANIATE_SHADER(getPositionFragmentShader)
INSTANIATE_SHADER(getPositionInstancedVertexShader)
INSTANIATE_SHADER(getPositionInstancedFragmentShader)
INSTANIATE_SHADER(getLinearDepthVertexShader)
INSTANIATE_SHADER(getLinearDepthFragmentShader)
INSTANIATE_SHADER(getLinearDepthInstancedVertexShader)
INSTANIATE_SHADER(getLinearDepthInstancedFragmentShader)
INSTANIATE_SHADER(getLineVertexShader)
INSTANIATE_SHADER(getLineFragmentShader)
INSTANIATE_SHADER(getGizmoVertexShader)
INSTANIATE_SHADER(getGizmoFragmentShader)
INSTANIATE_SHADER(getGizmoInstancedVertexShader)
INSTANIATE_SHADER(getGizmoInstancedFragmentShader)
INSTANIATE_SHADER(getGridVertexShader)
INSTANIATE_SHADER(getGridFragmentShader)
INSTANIATE_SHADER(getStandardVertexShader)
INSTANIATE_SHADER(getStandardFragmentShader)

// std::string PhysicsEngine::getGeometryVertexShader()
//{
//     switch (RenderContext::getRenderAPI())
//     {
//     case RenderAPI::OpenGL:
//         return glsl::getGeometryVertexShader();
//     case RenderAPI::DirectX:
//         return hlsl::getGeometryVertexShader();
//     }
// }

// std::string PhysicsEngine::getGeometryFragmentShader()
//{
//     switch (RenderContext::getRenderAPI())
//     {
//     case RenderAPI::OpenGL:
//         return glsl::getGeometryFragmentShader();
//     case RenderAPI::DirectX:
//         return hlsl::getGeometryFragmentShader();
//     }
// }
//
// std::string PhysicsEngine::getSSAOVertexShader()
//{
//     switch (RenderContext::getRenderAPI())
//     {
//     case RenderAPI::OpenGL:
//         return glsl::getSSAOVertexShader();
//     case RenderAPI::DirectX:
//         return hlsl::getSSAOVertexShader();
//     }
// }
//
// std::string PhysicsEngine::getSSAOFragmentShader()
//{
//     switch (RenderContext::getRenderAPI())
//     {
//     case RenderAPI::OpenGL:
//         return glsl::getSSAOFragmentShader();
//     case RenderAPI::DirectX:
//         return hlsl::getSSAOFragmentShader();
//     }
// }
//
// std::string PhysicsEngine::getShadowDepthMapVertexShader()
//{
//     switch (RenderContext::getRenderAPI())
//     {
//     case RenderAPI::OpenGL:
//         return glsl::getShadowDepthMapVertexShader();
//     case RenderAPI::DirectX:
//         return hlsl::getShadowDepthMapVertexShader();
//     }
// }
//
// std::string PhysicsEngine::getShadowDepthMapFragmentShader()
//{
//     switch (RenderContext::getRenderAPI())
//     {
//     case RenderAPI::OpenGL:
//         return glsl::getShadowDepthMapFragmentShader();
//     case RenderAPI::DirectX:
//         return hlsl::getShadowDepthMapFragmentShader();
//     }
// }
//
// std::string PhysicsEngine::getShadowDepthCubemapVertexShader()
//{
//     switch (RenderContext::getRenderAPI())
//     {
//     case RenderAPI::OpenGL:
//         return glsl::getShadowDepthCubemapVertexShader();
//     case RenderAPI::DirectX:
//         return hlsl::getShadowDepthCubemapVertexShader();
//     }
// }
//
// std::string PhysicsEngine::getShadowDepthCubemapFragmentShader()
//{
//     switch (RenderContext::getRenderAPI())
//     {
//     case RenderAPI::OpenGL:
//         return glsl::getShadowDepthCubemapFragmentShader();
//     case RenderAPI::DirectX:
//         return hlsl::getShadowDepthCubemapFragmentShader();
//     }
// }
//
// std::string PhysicsEngine::getShadowDepthCubemapGeometryShader()
//{
//     switch (RenderContext::getRenderAPI())
//     {
//     case RenderAPI::OpenGL:
//         return glsl::getShadowDepthCubemapGeometryShader();
//     case RenderAPI::DirectX:
//         return hlsl::getShadowDepthCubemapGeometryShader();
//     }
// }
//
// std::string PhysicsEngine::getColorVertexShader()
//{
//     switch (RenderContext::getRenderAPI())
//     {
//     case RenderAPI::OpenGL:
//         return glsl::getColorVertexShader();
//     case RenderAPI::DirectX:
//         return hlsl::getColorVertexShader();
//     }
// }
//
// std::string PhysicsEngine::getColorFragmentShader()
//{
//     switch (RenderContext::getRenderAPI())
//     {
//     case RenderAPI::OpenGL:
//         return glsl::getColorFragmentShader();
//     case RenderAPI::DirectX:
//         return hlsl::getColorFragmentShader();
//     }
// }
//
// std::string PhysicsEngine::getColorInstancedVertexShader()
//{
//     switch (RenderContext::getRenderAPI())
//     {
//     case RenderAPI::OpenGL:
//         return glsl::getColorInstancedVertexShader();
//     case RenderAPI::DirectX:
//         return hlsl::getColorInstancedVertexShader();
//     }
// }
//
// std::string PhysicsEngine::getColorInstancedFragmentShader()
//{
//     switch (RenderContext::getRenderAPI())
//     {
//     case RenderAPI::OpenGL:
//         return glsl::getColorInstancedFragmentShader();
//     case RenderAPI::DirectX:
//         return hlsl::getColorInstancedFragmentShader();
//     }
// }
//
// std::string PhysicsEngine::getScreenQuadVertexShader()
//{
//     switch (RenderContext::getRenderAPI())
//     {
//     case RenderAPI::OpenGL:
//         return glsl::getScreenQuadVertexShader();
//     case RenderAPI::DirectX:
//         return hlsl::getScreenQuadVertexShader();
//     }
// }
//
// std::string PhysicsEngine::getScreenQuadFragmentShader()
//{
//     switch (RenderContext::getRenderAPI())
//     {
//     case RenderAPI::OpenGL:
//         return glsl::getScreenQuadFragmentShader();
//     case RenderAPI::DirectX:
//         return hlsl::getScreenQuadFragmentShader();
//     }
// }
//
// std::string PhysicsEngine::getSpriteVertexShader()
//{
//     switch (RenderContext::getRenderAPI())
//     {
//     case RenderAPI::OpenGL:
//         return glsl::getSpriteVertexShader();
//     case RenderAPI::DirectX:
//         return hlsl::getSpriteVertexShader();
//     }
// }
//
// std::string PhysicsEngine::getSpriteFragmentShader()
//{
//     switch (RenderContext::getRenderAPI())
//     {
//     case RenderAPI::OpenGL:
//         return glsl::getSpriteFragmentShader();
//     case RenderAPI::DirectX:
//         return hlsl::getSpriteFragmentShader();
//     }
// }
//
// std::string PhysicsEngine::getGBufferVertexShader()
//{
//     switch (RenderContext::getRenderAPI())
//     {
//     case RenderAPI::OpenGL:
//         return glsl::getGBufferVertexShader();
//     case RenderAPI::DirectX:
//         return hlsl::getGBufferVertexShader();
//     }
// }
//
// std::string PhysicsEngine::getGBufferFragmentShader()
//{
//     switch (RenderContext::getRenderAPI())
//     {
//     case RenderAPI::OpenGL:
//         return glsl::getGBufferFragmentShader();
//     case RenderAPI::DirectX:
//         return hlsl::getGBufferFragmentShader();
//     }
// }
//
// std::string PhysicsEngine::getNormalVertexShader()
//{
//     switch (RenderContext::getRenderAPI())
//     {
//     case RenderAPI::OpenGL:
//         return glsl::getNormalVertexShader();
//     case RenderAPI::DirectX:
//         return hlsl::getNormalVertexShader();
//     }
// }
//
// std::string PhysicsEngine::getNormalFragmentShader()
//{
//     switch (RenderContext::getRenderAPI())
//     {
//     case RenderAPI::OpenGL:
//         return glsl::getNormalFragmentShader();
//     case RenderAPI::DirectX:
//         return hlsl::getNormalFragmentShader();
//     }
// }
//
// std::string PhysicsEngine::getNormalInstancedVertexShader()
//{
//     switch (RenderContext::getRenderAPI())
//     {
//     case RenderAPI::OpenGL:
//         return glsl::getNormalInstancedVertexShader();
//     case RenderAPI::DirectX:
//         return hlsl::getNormalInstancedVertexShader();
//     }
// }
//
// std::string PhysicsEngine::getNormalInstancedFragmentShader()
//{
//     switch (RenderContext::getRenderAPI())
//     {
//     case RenderAPI::OpenGL:
//         return glsl::getNormalInstancedFragmentShader();
//     case RenderAPI::DirectX:
//         return hlsl::getNormalInstancedFragmentShader();
//     }
// }
//
// std::string PhysicsEngine::getPositionVertexShader()
//{
//     switch (RenderContext::getRenderAPI())
//     {
//     case RenderAPI::OpenGL:
//         return glsl::getPositionVertexShader();
//     case RenderAPI::DirectX:
//         return hlsl::getPositionVertexShader();
//     }
// }
//
// std::string PhysicsEngine::getPositionFragmentShader()
//{
//     switch (RenderContext::getRenderAPI())
//     {
//     case RenderAPI::OpenGL:
//         return glsl::getPositionFragmentShader();
//     case RenderAPI::DirectX:
//         return hlsl::getPositionFragmentShader();
//     }
// }
//
// std::string PhysicsEngine::getPositionInstancedVertexShader()
//{
//     switch (RenderContext::getRenderAPI())
//     {
//     case RenderAPI::OpenGL:
//         return glsl::getPositionInstancedVertexShader();
//     case RenderAPI::DirectX:
//         return hlsl::getPositionInstancedVertexShader();
//     }
// }
//
// std::string PhysicsEngine::getPositionInstancedFragmentShader()
//{
//     switch (RenderContext::getRenderAPI())
//     {
//     case RenderAPI::OpenGL:
//         return glsl::getPositionInstancedFragmentShader();
//     case RenderAPI::DirectX:
//         return hlsl::getPositionInstancedFragmentShader();
//     }
// }
//
// std::string PhysicsEngine::getLinearDepthVertexShader()
//{
//     switch (RenderContext::getRenderAPI())
//     {
//     case RenderAPI::OpenGL:
//         return glsl::getLinearDepthVertexShader();
//     case RenderAPI::DirectX:
//         return hlsl::getLinearDepthVertexShader();
//     }
// }
//
// std::string PhysicsEngine::getLinearDepthFragmentShader()
//{
//     switch (RenderContext::getRenderAPI())
//     {
//     case RenderAPI::OpenGL:
//         return glsl::getLinearDepthFragmentShader();
//     case RenderAPI::DirectX:
//         return hlsl::getLinearDepthFragmentShader();
//     }
// }
//
// std::string PhysicsEngine::getLinearDepthInstancedVertexShader()
//{
//     switch (RenderContext::getRenderAPI())
//     {
//     case RenderAPI::OpenGL:
//         return glsl::getLinearDepthInstancedVertexShader();
//     case RenderAPI::DirectX:
//         return hlsl::getLinearDepthInstancedVertexShader();
//     }
// }
//
// std::string PhysicsEngine::getLinearDepthInstancedFragmentShader()
//{
//     switch (RenderContext::getRenderAPI())
//     {
//     case RenderAPI::OpenGL:
//         return glsl::getLinearDepthInstancedFragmentShader();
//     case RenderAPI::DirectX:
//         return hlsl::getLinearDepthInstancedFragmentShader();
//     }
// }
//
// std::string PhysicsEngine::getLineVertexShader()
//{
//     switch (RenderContext::getRenderAPI())
//     {
//     case RenderAPI::OpenGL:
//         return glsl::getLineVertexShader();
//     case RenderAPI::DirectX:
//         return hlsl::getLineVertexShader();
//     }
// }
//
// std::string PhysicsEngine::getLineFragmentShader()
//{
//     switch (RenderContext::getRenderAPI())
//     {
//     case RenderAPI::OpenGL:
//         return glsl::getLineFragmentShader();
//     case RenderAPI::DirectX:
//         return hlsl::getLineFragmentShader();
//     }
// }
//
// std::string PhysicsEngine::getGizmoVertexShader()
//{
//     switch (RenderContext::getRenderAPI())
//     {
//     case RenderAPI::OpenGL:
//         return glsl::getGizmoVertexShader();
//     case RenderAPI::DirectX:
//         return hlsl::getGizmoVertexShader();
//     }
// }
//
// std::string PhysicsEngine::getGizmoFragmentShader()
//{
//     switch (RenderContext::getRenderAPI())
//     {
//     case RenderAPI::OpenGL:
//         return glsl::getGizmoFragmentShader();
//     case RenderAPI::DirectX:
//         return hlsl::getGizmoFragmentShader();
//     }
// }
//
// std::string PhysicsEngine::getGridVertexShader()
//{
//     switch (RenderContext::getRenderAPI())
//     {
//     case RenderAPI::OpenGL:
//         return glsl::getGridVertexShader();
//     case RenderAPI::DirectX:
//         return hlsl::getGridVertexShader();
//     }
// }
//
// std::string PhysicsEngine::getGridFragmentShader()
//{
//     switch (RenderContext::getRenderAPI())
//     {
//     case RenderAPI::OpenGL:
//         return glsl::getGridFragmentShader();
//     case RenderAPI::DirectX:
//         return hlsl::getGridFragmentShader();
//     }
// }
//
// std::string PhysicsEngine::getStandardVertexShader()
//{
//     switch (RenderContext::getRenderAPI())
//     {
//     case RenderAPI::OpenGL:
//         return glsl::getStandardVertexShader();
//     case RenderAPI::DirectX:
//         return hlsl::getStandardVertexShader();
//     }
// }
//
// std::string PhysicsEngine::getStandardFragmentShader()
//{
//     switch (RenderContext::getRenderAPI())
//     {
//     case RenderAPI::OpenGL:
//         return glsl::getStandardFragmentShader();
//     case RenderAPI::DirectX:
//         return hlsl::getStandardFragmentShader();
//     }
// }