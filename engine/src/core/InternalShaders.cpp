#include "../../include/core/InternalShaders.h"

using namespace PhysicsEngine;

#include "shaders/standard.vs"
#include "shaders/standard.fs"
#include "shaders/colorLit.vs"
#include "shaders/colorLit.fs"
#include "shaders/normal.vs"
#include "shaders/normal.fs"
#include "shaders/tangent.vs"
#include "shaders/tangent.fs"
#include "shaders/binormal.vs"
#include "shaders/binormal.fs"
#include "shaders/gizmo.vs"
#include "shaders/gizmo.fs"
#include "shaders/color.vs"
#include "shaders/color.fs"
#include "shaders/line.vs"
#include "shaders/line.fs"
#include "shaders/screenQuad.vs"
#include "shaders/screenQuad.fs"
#include "shaders/normalMap.vs"
#include "shaders/normalMap.fs"
#include "shaders/depthMap.vs"
#include "shaders/depthMap.fs"
#include "shaders/shadowDepthMap.vs"
#include "shaders/shadowDepthMap.fs"
#include "shaders/shadowDepthCubemap.vs"
#include "shaders/shadowDepthCubemap.gs"
#include "shaders/shadowDepthCubemap.fs"
#include "shaders/gbuffer.vs"
#include "shaders/gbuffer.fs"
#include "shaders/positionAndNormals.vs"
#include "shaders/positionAndNormals.fs"
#include "shaders/ssao.vs"
#include "shaders/ssao.fs"
#include "shaders/standardDeferred.vs"
#include "shaders/standardDeferred.fs"
#include "shaders/grid.vs"
#include "shaders/grid.fs"

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
const std::string InternalShaders::gridShaderName = "Grid";

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