#include "../../include/core/InternalShaders.h"

using namespace PhysicsEngine;

#define STRINGIFY(x) #x

const std::string InternalShaders::standardVertexShader =
#include "shaders/standard.vs"
;

const std::string InternalShaders::standardFragmentShader =
#include "shaders/standard.fs"
;

const std::string InternalShaders::colorLitVertexShader =
#include "shaders/colorLit.vs"
;

const std::string InternalShaders::colorLitFragmentShader =
#include "shaders/colorLit.fs"
;

const std::string InternalShaders::normalVertexShader =
#include "shaders/normal.vs"
;

const std::string InternalShaders::normalFragmentShader =
#include "shaders/normal.fs"
;

const std::string InternalShaders::tangentVertexShader =
#include "shaders/tangent.vs"
;

const std::string InternalShaders::tangentFragmentShader =
#include "shaders/tangent.fs"
;

const std::string InternalShaders::binormalVertexShader =
#include "shaders/binormal.vs"
;

const std::string InternalShaders::binormalFragmentShader =
#include "shaders/binormal.fs"
;

const std::string InternalShaders::gizmoVertexShader =
#include "shaders/gizmo.vs"
;

const std::string InternalShaders::gizmoFragmentShader =
#include "shaders/gizmo.fs"
;

const std::string InternalShaders::colorVertexShader =
#include "shaders/color.vs"
;

const std::string InternalShaders::colorFragmentShader =
#include "shaders/color.fs"
;

const std::string InternalShaders::lineVertexShader =
#include "shaders/line.vs"
;

const std::string InternalShaders::lineFragmentShader =
#include "shaders/line.fs"
;

const std::string InternalShaders::screenQuadVertexShader =
#include "shaders/screenQuad.vs"
;

const std::string InternalShaders::screenQuadFragmentShader =
#include "shaders/screenQuad.fs"
;

const std::string InternalShaders::normalMapVertexShader =
#include "shaders/normalMap.vs"
;

const std::string InternalShaders::normalMapFragmentShader =
#include "shaders/normalMap.fs"
;

const std::string InternalShaders::depthMapVertexShader =
#include "shaders/depthMap.vs"
;

const std::string InternalShaders::depthMapFragmentShader =
#include "shaders/depthMap.fs"
;

const std::string InternalShaders::shadowDepthMapVertexShader =
#include "shaders/shadowDepthMap.vs"
;

const std::string InternalShaders::shadowDepthMapFragmentShader =
#include "shaders/shadowDepthMap.fs"
;

const std::string InternalShaders::shadowDepthCubemapVertexShader =
#include "shaders/shadowDepthCubemap.vs"
;

const std::string InternalShaders::shadowDepthCubemapGeometryShader =
#include "shaders/shadowDepthCubemap.gs"
;

const std::string InternalShaders::shadowDepthCubemapFragmentShader =
#include "shaders/shadowDepthCubemap.fs"
;

const std::string InternalShaders::gbufferVertexShader =
#include "shaders/gbuffer.vs"
;

const std::string InternalShaders::gbufferFragmentShader =
#include "shaders/gbuffer.fs"
;

const std::string InternalShaders::positionAndNormalsVertexShader =
#include "shaders/positionAndNormals.vs"
;

const std::string InternalShaders::positionAndNormalsFragmentShader =
#include "shaders/positionAndNormals.fs"
;

const std::string InternalShaders::ssaoVertexShader =
#include "shaders/ssao.vs"
;

const std::string InternalShaders::ssaoFragmentShader =
#include "shaders/ssao.fs"
;

const std::string InternalShaders::standardDeferredVertexShader =
#include "shaders/standardDeferred.vs"
;

const std::string InternalShaders::standardDeferredFragmentShader =
#include "shaders/standardDeferred.fs"
;

const std::string InternalShaders::standardShaderName = "Standard";
const std::string InternalShaders::colorLitShaderName = "ColorLit";
const std::string InternalShaders::normalShaderName = "Normal";
const std::string InternalShaders::tangentShaderName = "Tangent";
const std::string InternalShaders::binormalShaderName = "Binormal";
const std::string InternalShaders::gizmoShaderName = "Gizmo";
const std::string InternalShaders::colorShaderName = "Color";
const std::string InternalShaders::lineShaderName = "Line";
const std::string InternalShaders::screenQuadShaderName = "ScreenQuad";
const std::string InternalShaders::normalMapShaderName = "NormalMap";
const std::string InternalShaders::depthMapShaderName = "DepthMap";
const std::string InternalShaders::shadowDepthMapShaderName = "ShadowDepthMap";
const std::string InternalShaders::shadowDepthCubemapShaderName = "ShadowDepthCubemap";
const std::string InternalShaders::gbufferShaderName = "GBuffer";
const std::string InternalShaders::positionAndNormalShaderName = "PositionAndNormal";
const std::string InternalShaders::ssaoShaderName = "SSAO";
const std::string InternalShaders::standardDeferredShaderName = "StandardDeferred";

Guid InternalShaders::loadInternalShader(World *world, const std::string &name,
                                         const std::string vertex, const std::string fragment,
                                         const std::string geometry)
{
    PhysicsEngine::Shader *shader = world->createAsset<PhysicsEngine::Shader>();
    if (shader != nullptr)
    {
        shader->setVertexShader(vertex);
        shader->setFragmentShader(fragment);
        shader->setGeometryShader(geometry);
        shader->setName(name);
        return shader->getId();
    }

    return Guid::INVALID;
}