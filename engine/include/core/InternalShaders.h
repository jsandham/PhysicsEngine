#ifndef INTERNAL_SHADERS_H__
#define INTERNAL_SHADERS_H__

#include <string>

#include "World.h"

namespace PhysicsEngine
{
class InternalShaders
{
  public:
    static const std::string standardVertexShader;
    static const std::string standardFragmentShader;
    static const std::string colorLitVertexShader;
    static const std::string colorLitFragmentShader;
    static const std::string normalVertexShader;
    static const std::string normalFragmentShader;
    static const std::string tangentVertexShader;
    static const std::string tangentFragmentShader;
    static const std::string binormalVertexShader;
    static const std::string binormalFragmentShader;
    static const std::string gizmoVertexShader;
    static const std::string gizmoFragmentShader;
    static const std::string colorVertexShader;
    static const std::string colorFragmentShader;
    static const std::string lineVertexShader;
    static const std::string lineFragmentShader;
    static const std::string screenQuadVertexShader;
    static const std::string screenQuadFragmentShader;
    static const std::string normalMapVertexShader;
    static const std::string normalMapFragmentShader;
    static const std::string depthMapVertexShader;
    static const std::string depthMapFragmentShader;
    static const std::string shadowDepthMapVertexShader;
    static const std::string shadowDepthMapFragmentShader;
    static const std::string shadowDepthCubemapVertexShader;
    static const std::string shadowDepthCubemapGeometryShader;
    static const std::string shadowDepthCubemapFragmentShader;
    static const std::string gbufferVertexShader;
    static const std::string gbufferFragmentShader;
    static const std::string positionAndNormalsVertexShader;
    static const std::string positionAndNormalsFragmentShader;
    static const std::string ssaoVertexShader;
    static const std::string ssaoFragmentShader;
    static const std::string standardDeferredVertexShader;
    static const std::string standardDeferredFragmentShader;
    static const std::string gridVertexShader;
    static const std::string gridFragmentShader;

    static const std::string standardShaderName;
    static const std::string colorLitShaderName;
    static const std::string normalShaderName;
    static const std::string tangentShaderName;
    static const std::string binormalShaderName;
    static const std::string gizmoShaderName;
    static const std::string colorShaderName;
    static const std::string lineShaderName;
    static const std::string screenQuadShaderName;
    static const std::string normalMapShaderName;
    static const std::string depthMapShaderName;
    static const std::string shadowDepthMapShaderName;
    static const std::string shadowDepthCubemapShaderName;
    static const std::string gbufferShaderName;
    static const std::string positionAndNormalShaderName;
    static const std::string ssaoShaderName;
    static const std::string standardDeferredShaderName;
    static const std::string gridShaderName;

    enum class Shader
    {
        Standard,
        ColorLit,
        Normal,
        Tangent,
        Binormal,
        Gizmo,
        Color,
        Line,
        ScreenQuad,
        NormalMap,
        DepthMap,
        ShadowDepthMap,
        ShadowDepthCubemap,
        GBuffer,
        PositionAndNormals,
        SSAO,
        StandardDeferred,
        Grid
    };

    template<Shader S>
    static Guid loadShader(World* world)
    {
        return Guid::INVALID;
    }

    template<>
    static Guid loadShader<Shader::Standard>(World* world)
    {
        return loadInternalShader(world, standardShaderName, standardVertexShader,
            standardFragmentShader, "");
    }

    template<>
    static Guid loadShader<Shader::ColorLit>(World* world)
    {
        return loadInternalShader(world, colorLitShaderName, colorLitVertexShader,
            colorLitFragmentShader, "");
    }

    template<>
    static Guid loadShader<Shader::Normal>(World* world)
    {
        return loadInternalShader(world, normalShaderName, normalVertexShader,
            normalFragmentShader, "");
    }

    template<>
    static Guid loadShader<Shader::Tangent>(World* world)
    {
        return loadInternalShader(world, tangentShaderName, tangentVertexShader,
            tangentFragmentShader, "");
    }

    template<>
    static Guid loadShader<Shader::Binormal>(World* world)
    {
        return loadInternalShader(world, binormalShaderName, binormalVertexShader,
            binormalFragmentShader, "");
    }

    template<>
    static Guid loadShader<Shader::Gizmo>(World* world)
    {
        return loadInternalShader(world, gizmoShaderName, gizmoVertexShader,
            gizmoFragmentShader, "");
    }

    template<>
    static Guid loadShader<Shader::Color>(World* world)
    {
        return loadInternalShader(world, colorShaderName, colorVertexShader,
            colorFragmentShader, "");
    }

    template<>
    static Guid loadShader<Shader::Line>(World* world)
    {
        return loadInternalShader(world, lineShaderName, lineVertexShader,
            lineFragmentShader, "");
    }

    template<>
    static Guid loadShader<Shader::ScreenQuad>(World* world)
    {
        return loadInternalShader(world, screenQuadShaderName, screenQuadVertexShader,
            screenQuadFragmentShader, "");
    }

    template<>
    static Guid loadShader<Shader::NormalMap>(World* world)
    {
        return loadInternalShader(world, normalMapShaderName, normalMapVertexShader,
            normalMapFragmentShader, "");
    }

    template<>
    static Guid loadShader<Shader::DepthMap>(World* world)
    {
        return loadInternalShader(world, depthMapShaderName, depthMapVertexShader,
            depthMapFragmentShader, "");
    }

    template<>
    static Guid loadShader<Shader::ShadowDepthMap>(World* world)
    {
        return loadInternalShader(world, shadowDepthMapShaderName, shadowDepthMapVertexShader,
            shadowDepthMapFragmentShader, "");
    }

    template<>
    static Guid loadShader<Shader::ShadowDepthCubemap>(World* world)
    {
        return loadInternalShader(world, shadowDepthCubemapShaderName, shadowDepthCubemapVertexShader,
            shadowDepthCubemapFragmentShader, shadowDepthCubemapGeometryShader);
    }

    template<>
    static Guid loadShader<Shader::GBuffer>(World* world)
    {
        return loadInternalShader(world, gbufferShaderName, gbufferVertexShader,
            gbufferFragmentShader, "");
    }

    template<>
    static Guid loadShader<Shader::PositionAndNormals>(World* world)
    {
        return loadInternalShader(world, positionAndNormalShaderName, positionAndNormalsVertexShader,
            positionAndNormalsFragmentShader, "");
    }

    template<>
    static Guid loadShader<Shader::SSAO>(World* world)
    {
        return loadInternalShader(world, ssaoShaderName, ssaoVertexShader,
            ssaoFragmentShader, "");
    }

    template<>
    static Guid loadShader<Shader::StandardDeferred>(World* world)
    {
        return loadInternalShader(world, standardDeferredShaderName, standardDeferredVertexShader,
            standardDeferredFragmentShader, "");
    }

    template<>
    static Guid loadShader<Shader::Grid>(World* world)
    {
        return loadInternalShader(world, gridShaderName, gridVertexShader, gridFragmentShader, "");
    }

  private:
    static Guid loadInternalShader(World *world, const std::string &name, const std::string vertex,
                                   const std::string fragment, const std::string geometry);
};
} // namespace PhysicsEngine

#endif