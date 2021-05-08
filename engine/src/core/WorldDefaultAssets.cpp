#include "../../include/core/WorldDefaultAssets.h"

#include "../../include/core/InternalMaterials.h"
#include "../../include/core/InternalMeshes.h"
#include "../../include/core/InternalShaders.h"

#include "../../include/core/World.h"

using namespace PhysicsEngine;

void WorldDefaultAssets::loadInternalAssets(World* world)
{
    // load default included meshes
    mSphereMeshId = InternalMeshes::loadMesh<InternalMeshes::Mesh::Sphere>(world);
    mCubeMeshId = InternalMeshes::loadMesh<InternalMeshes::Mesh::Cube>(world);
    mPlaneMeshId = InternalMeshes::loadMesh<InternalMeshes::Mesh::Plane>(world);

    // load default included shaders
    mStandardShaderId = InternalShaders::loadShader<InternalShaders::Shader::Standard>(world);
    mColorLitShaderId = InternalShaders::loadShader<InternalShaders::Shader::ColorLit>(world);
    mNormalShaderId = InternalShaders::loadShader<InternalShaders::Shader::Normal>(world);
    mTangentShaderId = InternalShaders::loadShader<InternalShaders::Shader::Tangent>(world);
    mBinormalShaderId = InternalShaders::loadShader<InternalShaders::Shader::Binormal>(world);
    mGizmoShaderId = InternalShaders::loadShader<InternalShaders::Shader::Gizmo>(world);
    mLineShaderId = InternalShaders::loadShader<InternalShaders::Shader::Line>(world);
    mColorShaderId = InternalShaders::loadShader<InternalShaders::Shader::Color>(world);
    mScreenQuadShaderId = InternalShaders::loadShader<InternalShaders::Shader::ScreenQuad>(world);
    mNormalMapShaderId = InternalShaders::loadShader<InternalShaders::Shader::NormalMap>(world);
    mDepthMapShaderId = InternalShaders::loadShader<InternalShaders::Shader::DepthMap>(world);
    mShadowDepthMapShaderId = InternalShaders::loadShader<InternalShaders::Shader::ShadowDepthMap>(world);
    mShadowDepthCubemapShaderId = InternalShaders::loadShader<InternalShaders::Shader::ShadowDepthCubemap>(world);
    mGbufferShaderId = InternalShaders::loadShader<InternalShaders::Shader::GBuffer>(world);
    mPositionAndNormalsShaderId = InternalShaders::loadShader<InternalShaders::Shader::PositionAndNormals>(world);
    mSsaoShaderId = InternalShaders::loadShader<InternalShaders::Shader::SSAO>(world);
    //mStandardDeferedShaderId = InternalShaders::loadShader<InternalShaders::Shader::StandardDeferred>(world);
    mGridShaderId = InternalShaders::loadShader<InternalShaders::Shader::Grid>(world);

    // load default included materials
    mSimpleLitMaterialId =
        InternalMaterials::loadMaterial<InternalMaterials::Material::SimpleLit>(world, mStandardShaderId);
    mColorMaterialId = InternalMaterials::loadMaterial<InternalMaterials::Material::Color>(world, mColorShaderId);
}