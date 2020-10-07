#ifndef __INTERNAL_SHADERS_H__
#define __INTERNAL_SHADERS_H__

#include <string>

#include "Guid.h"
#include "Shader.h"

namespace PhysicsEngine
{
class InternalShaders
{
  public:
    static const std::string gizmoVertexShader;
    static const std::string gizmoFragmentShader;

    static const std::string fontVertexShader;
    static const std::string fontFragmentShader;
    static const std::string colorVertexShader;
    static const std::string colorFragmentShader;
    static const std::string positionAndNormalsVertexShader;
    static const std::string positionAndNormalsFragmentShader;
    static const std::string ssaoVertexShader;
    static const std::string ssaoFragmentShader;
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
    static const std::string simpleLitVertexShader;
    static const std::string simpleLitFragmentShader;
    static const std::string simpleLitDeferredVertexShader;
    static const std::string simpleLitDeferredFragmentShader;
    static const std::string overdrawVertexShader;
    static const std::string overdrawFragmentShader;

    static const Guid fontShaderId;
    static const Guid colorShaderId;
    static const Guid positionAndNormalShaderId;
    static const Guid ssaoShaderId;
    static const Guid screenQuadShaderId;
    static const Guid normalMapShaderId;
    static const Guid depthMapShaderId;
    static const Guid shadowDepthMapShaderId;
    static const Guid shadowDepthCubemapShaderId;
    static const Guid gbufferShaderId;
    static const Guid simpleLitShaderId;
    static const Guid simpleLitDeferredShaderId;
    static const Guid overdrawShaderId;

    static Guid loadFontShader(World *world);
    static Guid loadColorShader(World *world);
    static Guid loadPositionAndNormalsShader(World *world);
    static Guid loadSsaoShader(World *world);
    static Guid loadScreenQuadShader(World *world);
    static Guid loadNormalMapShader(World *world);
    static Guid loadDepthMapShader(World *world);
    static Guid loadShadowDepthMapShader(World *world);
    static Guid loadShadowDepthCubemapShader(World *world);
    static Guid loadGBufferShader(World *world);
    static Guid loadSimpleLitShader(World *world);
    static Guid loadSimpleLitDeferredShader(World *world);
    static Guid loadOverdrawShader(World *world);

  private:
    static Guid loadInternalShader(World *world, const Guid shaderId, const std::string vertex,
                                   const std::string fragment, const std::string geometry);
};
} // namespace PhysicsEngine

#endif