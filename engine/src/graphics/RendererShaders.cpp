#include "../../include/core/Log.h"
#include "../../include/core/Shader.h"

#include "../../include/graphics/InternalShaders.h"
#include "../../include/graphics/RendererShaders.h"

using namespace PhysicsEngine;

StandardShader::StandardShader()
{
    mShader = ShaderProgram::create();
    mShader->load("Standard", getStandardVertexShader(), getStandardFragmentShader());
    mShader->compile();
}

StandardShader::~StandardShader()
{
    delete mShader;
}

void StandardShader::bind()
{
    mShader->bind();
}

void StandardShader::unbind()
{
    mShader->unbind();
}

std::string StandardShader::getVertexShader() const
{
    return mShader->getVertexShader();
}

std::string StandardShader::getFragmentShader() const
{
    return mShader->getFragmentShader();
}

GBufferShader::GBufferShader()
{
    mShader = ShaderProgram::create();
    mShader->load("GBuffer", getGBufferVertexShader(), getGBufferFragmentShader());
    mShader->compile();

    mModelId = Shader::uniformToId("model");
}

GBufferShader ::~GBufferShader()
{
    delete mShader;
}

void GBufferShader::bind()
{
    mShader->bind();
}

void GBufferShader::unbind()
{
    mShader->unbind();
}

void GBufferShader::setModel(const glm::mat4 &model)
{
    mShader->setMat4(mModelId, model);
}

QuadShader::QuadShader()
{
    mShader = ShaderProgram::create();
    mShader->load("ScreenQuad", getScreenQuadVertexShader(), getScreenQuadFragmentShader());
    mShader->compile();

    mScreenTexId = Shader::uniformToId("screenTexture");
}

QuadShader ::~QuadShader()
{
    delete mShader;
}

void QuadShader::bind()
{
    mShader->bind();
}

void QuadShader::unbind()
{
    mShader->unbind();
}

void QuadShader::setScreenTexture(int texUnit, void *tex)
{
    mShader->setTexture2D(mScreenTexId, texUnit, tex);
}

DepthShader::DepthShader()
{
    mShader = ShaderProgram::create();
    mShader->load("DepthMap", getShadowDepthMapVertexShader(), getShadowDepthMapFragmentShader());
    mShader->compile();

    mModelId = Shader::uniformToId("model");
    mViewId = Shader::uniformToId("view");
    mProjectionId = Shader::uniformToId("projection");
}

DepthShader ::~DepthShader()
{
    delete mShader;
}

void DepthShader::bind()
{
    mShader->bind();
}

void DepthShader::unbind()
{
    mShader->unbind();
}

void DepthShader::setModel(const glm::mat4 &model)
{
    mShader->setMat4(mModelId, model);
}

void DepthShader::setView(const glm::mat4 &view)
{
    mShader->setMat4(mViewId, view);
}

void DepthShader::setProjection(const glm::mat4 &projection)
{
    mShader->setMat4(mProjectionId, projection);
}

DepthCubemapShader::DepthCubemapShader()
{
    mShader = ShaderProgram::create();
    mShader->load("DepthCubemap", getShadowDepthCubemapVertexShader(), getShadowDepthCubemapFragmentShader(),
                  getShadowDepthCubemapGeometryShader());
    mShader->compile();

    mLightPosId = Shader::uniformToId("lightPos");
    mFarPlaneId = Shader::uniformToId("farPlane");
    mModelId = Shader::uniformToId("model");
    mCubeViewProjMatricesId[0] = Shader::uniformToId("cubeViewProjMatrices[0]");
    mCubeViewProjMatricesId[1] = Shader::uniformToId("cubeViewProjMatrices[1]");
    mCubeViewProjMatricesId[2] = Shader::uniformToId("cubeViewProjMatrices[2]");
    mCubeViewProjMatricesId[3] = Shader::uniformToId("cubeViewProjMatrices[3]");
    mCubeViewProjMatricesId[4] = Shader::uniformToId("cubeViewProjMatrices[4]");
    mCubeViewProjMatricesId[5] = Shader::uniformToId("cubeViewProjMatrices[5]");
}

DepthCubemapShader::~DepthCubemapShader()
{
    delete mShader;
}

void DepthCubemapShader::bind()
{
    mShader->bind();
}

void DepthCubemapShader::unbind()
{
    mShader->unbind();
}

void DepthCubemapShader::setLightPos(const glm::vec3 &lightPos)
{
    mShader->setVec3(mLightPosId, lightPos);
}

void DepthCubemapShader::setFarPlane(float farPlane)
{
    mShader->setFloat(mFarPlaneId, farPlane);
}

void DepthCubemapShader::setModel(const glm::mat4 &model)
{
    mShader->setMat4(mModelId, model);
}

void DepthCubemapShader::setCubeViewProj(int index, const glm::mat4 &modelView)
{
    assert(index >= 0);
    assert(index <= 5);
    mShader->setMat4(mCubeViewProjMatricesId[index], modelView);
}

GeometryShader::GeometryShader()
{
    mShader = ShaderProgram::create();
    mShader->load("Geometry", getGeometryVertexShader(), getGeometryFragmentShader());
    mShader->compile();

    mModelId = Shader::uniformToId("model");
}

GeometryShader::~GeometryShader()
{
    delete mShader;
}

void GeometryShader::bind()
{
    mShader->bind();
}

void GeometryShader::unbind()
{
    mShader->unbind();
}

void GeometryShader::setModel(const glm::mat4 &model)
{
    mShader->setMat4(mModelId, model);
}

NormalShader::NormalShader()
{
    mShader = ShaderProgram::create();
    mShader->load("Normal", getNormalVertexShader(), getNormalFragmentShader());
    mShader->compile();

    mModelId = Shader::uniformToId("model");
}

NormalShader::~NormalShader()
{
    delete mShader;
}

void NormalShader::bind()
{
    mShader->bind();
}

void NormalShader::unbind()
{
    mShader->unbind();
}

void NormalShader::setModel(const glm::mat4 &model)
{
    mShader->setMat4(mModelId, model);
}

NormalInstancedShader::NormalInstancedShader()
{
    mShader = ShaderProgram::create();
    mShader->load("Normal Instanced", getNormalInstancedVertexShader(), getNormalInstancedFragmentShader());
    mShader->compile();
}

NormalInstancedShader::~NormalInstancedShader()
{
    delete mShader;
}

void NormalInstancedShader::bind()
{
    mShader->bind();
}

void NormalInstancedShader::unbind()
{
    mShader->unbind();
}

PositionShader::PositionShader()
{
    mShader = ShaderProgram::create();
    mShader->load("Position", getPositionVertexShader(), getPositionFragmentShader());
    mShader->compile();

    mModelId = Shader::uniformToId("model");
}

PositionShader::~PositionShader()
{
    delete mShader;
}

void PositionShader::bind()
{
    mShader->bind();
}

void PositionShader::unbind()
{
    mShader->unbind();
}

void PositionShader::setModel(const glm::mat4 &model)
{
    mShader->setMat4(mModelId, model);
}

PositionInstancedShader::PositionInstancedShader()
{
    mShader = ShaderProgram::create();
    mShader->load("Position Instanced", getPositionInstancedVertexShader(), getPositionInstancedFragmentShader());
    mShader->compile();
}

PositionInstancedShader::~PositionInstancedShader()
{
    delete mShader;
}

void PositionInstancedShader::bind()
{
    mShader->bind();
}

void PositionInstancedShader::unbind()
{
    mShader->unbind();
}

LinearDepthShader::LinearDepthShader()
{
    mShader = ShaderProgram::create();
    mShader->load("LinearDepth", getLinearDepthVertexShader(), getLinearDepthFragmentShader());
    mShader->compile();

    mModelId = Shader::uniformToId("model");
}

LinearDepthShader::~LinearDepthShader()
{
    delete mShader;
}

void LinearDepthShader::bind()
{
    mShader->bind();
}

void LinearDepthShader::unbind()
{
    mShader->unbind();
}

void LinearDepthShader::setModel(const glm::mat4 &model)
{
    mShader->setMat4(mModelId, model);
}

LinearDepthInstancedShader::LinearDepthInstancedShader()
{
    mShader = ShaderProgram::create();
    mShader->load("Linear Depth Instanced", getLinearDepthInstancedVertexShader(),
                  getLinearDepthInstancedFragmentShader());
    mShader->compile();
}

LinearDepthInstancedShader::~LinearDepthInstancedShader()
{
    delete mShader;
}

void LinearDepthInstancedShader::bind()
{
    mShader->bind();
}

void LinearDepthInstancedShader::unbind()
{
    mShader->unbind();
}

ColorShader::ColorShader()
{
    mShader = ShaderProgram::create();
    mShader->load("Color", getColorVertexShader(), getColorFragmentShader());
    mShader->compile();

    mModelId = Shader::uniformToId("model");
    mColorId = Shader::uniformToId("material.color");
}

ColorShader::~ColorShader()
{
    delete mShader;
}

void ColorShader::bind()
{
    mShader->bind();
}

void ColorShader::unbind()
{
    mShader->unbind();
}

void ColorShader::setModel(const glm::mat4 &model)
{
    mShader->setMat4(mModelId, model);
}

void ColorShader::setColor32(const Color32 &color)
{
    mShader->setColor32(mColorId, color);
}

ColorInstancedShader::ColorInstancedShader()
{
    mShader = ShaderProgram::create();
    mShader->load("Color Instanced", getColorInstancedVertexShader(), getColorInstancedFragmentShader());
    mShader->compile();
}

ColorInstancedShader::~ColorInstancedShader()
{
    delete mShader;
}

void ColorInstancedShader::bind()
{
    mShader->bind();
}

void ColorInstancedShader::unbind()
{
    mShader->unbind();
}

SSAOShader::SSAOShader()
{
    mShader = ShaderProgram::create();
    mShader->load("SSAO", getSSAOVertexShader(), getSSAOFragmentShader());
    mShader->compile();

    mProjectionId = Shader::uniformToId("projection");
    mPositionTexId = Shader::uniformToId("positionTex");
    mNormalTexId = Shader::uniformToId("normalTex");
    mNoiseTexId = Shader::uniformToId("noiseTex");
    for (int i = 0; i < 64; i++)
    {
        mSamplesId[i] = Shader::uniformToId(("samples[" + std::to_string(i) + "]").c_str());
    }
}

SSAOShader::~SSAOShader()
{
    delete mShader;
}

void SSAOShader::bind()
{
    mShader->bind();
}

void SSAOShader::unbind()
{
    mShader->unbind();
}

void SSAOShader::setProjection(const glm::mat4 &projection)
{
    mShader->setMat4(mProjectionId, projection);
}

void SSAOShader::setPositionTexture(int texUnit, void *tex)
{
    mShader->setTexture2D(mNormalTexId, texUnit, tex);
}

void SSAOShader::setNormalTexture(int texUnit, void *tex)
{
    mShader->setTexture2D(mPositionTexId, texUnit, tex);
}

void SSAOShader::setNoiseTexture(int texUnit, void *tex)
{
    mShader->setTexture2D(mNoiseTexId, texUnit, tex);
}

void SSAOShader::setSample(int index, const glm::vec3 &sample)
{
    assert(index >= 0);
    assert(index <= 63);
    mShader->setVec3(mSamplesId[index], sample);
}

SpriteShader::SpriteShader()
{
    mShader = ShaderProgram::create();
    mShader->load("Sprite", getSpriteVertexShader(), getSpriteFragmentShader());
    mShader->compile();

    mModelId = Shader::uniformToId("model");
    mViewId = Shader::uniformToId("view");
    mProjectionId = Shader::uniformToId("projection");
    mColorId = Shader::uniformToId("spriteColor");
    mImageId = Shader::uniformToId("image");
}

SpriteShader::~SpriteShader()
{
    delete mShader;
}

void SpriteShader::bind()
{
    mShader->bind();
}

void SpriteShader::unbind()
{
    mShader->unbind();
}

void SpriteShader::setModel(const glm::mat4 &model)
{
    mShader->setMat4(mModelId, model);
}

void SpriteShader::setView(const glm::mat4 &view)
{
    mShader->setMat4(mViewId, view);
}

void SpriteShader::setProjection(const glm::mat4 &projection)
{
    mShader->setMat4(mProjectionId, projection);
}

void SpriteShader::setColor(const Color &color)
{
    mShader->setColor(mColorId, color);
}

void SpriteShader::setImage(int texUnit, void *tex)
{
    mShader->setTexture2D(mImageId, texUnit, tex);
}

LineShader::LineShader()
{
    mShader = ShaderProgram::create();
    mShader->load("Line", getLineVertexShader(), getLineFragmentShader());
    mShader->compile();

    mMVPId = Shader::uniformToId("mvp");
}

LineShader::~LineShader()
{
    delete mShader;
}

void LineShader::bind()
{
    mShader->bind();
}

void LineShader::unbind()
{
    mShader->unbind();
}

void LineShader::setMVP(const glm::mat4 &mvp)
{
    mShader->setMat4(mMVPId, mvp);
}

GizmoShader::GizmoShader()
{
    mShader = ShaderProgram::create();
    mShader->load("Gizmo", getGizmoVertexShader(), getGizmoFragmentShader());
    mShader->compile();

    mModelId = Shader::uniformToId("model");
    mViewId = Shader::uniformToId("view");
    mProjectionId = Shader::uniformToId("projection");
    mColorId = Shader::uniformToId("color");
    mLightPosId = Shader::uniformToId("lightPos");
}

GizmoShader::~GizmoShader()
{
    delete mShader;
}

void GizmoShader::bind()
{
    mShader->bind();
}

void GizmoShader::unbind()
{
    mShader->unbind();
}

void GizmoShader::setModel(const glm::mat4 &model)
{
    mShader->setMat4(mModelId, model);
}

void GizmoShader::setView(const glm::mat4 &view)
{
    mShader->setMat4(mViewId, view);
}

void GizmoShader::setProjection(const glm::mat4 &projection)
{
    mShader->setMat4(mProjectionId, projection);
}

void GizmoShader::setColor(const Color &color)
{
    mShader->setColor(mColorId, color);
}

void GizmoShader::setLightPos(const glm::vec3 &lightPos)
{
    mShader->setVec3(mLightPosId, lightPos);
}

GridShader::GridShader()
{
    mShader = ShaderProgram::create();
    mShader->load("Grid", getGridVertexShader(), getGridFragmentShader());
    mShader->compile();

    mMVPId = Shader::uniformToId("mvp");
    mColorId = Shader::uniformToId("color");
}

GridShader::~GridShader()
{
    delete mShader;
}

void GridShader::bind()
{
    mShader->bind();
}

void GridShader::unbind()
{
    mShader->unbind();
}

void GridShader::setMVP(const glm::mat4 &mvp)
{
    mShader->setMat4(mMVPId, mvp);
}

void GridShader::setColor(const Color &color)
{
    mShader->setColor(mColorId, color);
}

StandardShader *RendererShaders::sStandardShader = nullptr;
SSAOShader *RendererShaders::sSSAOShader = nullptr;
GeometryShader *RendererShaders::sGeometryShader = nullptr;
DepthShader *RendererShaders::sDepthShader = nullptr;
DepthCubemapShader *RendererShaders::sDepthCubemapShader = nullptr;
QuadShader *RendererShaders::sQuadShader = nullptr;
SpriteShader *RendererShaders::sSpriteShader = nullptr;
GBufferShader *RendererShaders::sGBufferShader = nullptr;
ColorShader *RendererShaders::sColorShader = nullptr;
ColorInstancedShader *RendererShaders::sColorInstancedShader = nullptr;
NormalShader *RendererShaders::sNormalShader = nullptr;
NormalInstancedShader *RendererShaders::sNormalInstancedShader = nullptr;
PositionShader *RendererShaders::sPositionShader = nullptr;
PositionInstancedShader *RendererShaders::sPositionInstancedShader = nullptr;
LinearDepthShader *RendererShaders::sLinearDepthShader = nullptr;
LinearDepthInstancedShader *RendererShaders::sLinearDepthInstancedShader = nullptr;
LineShader *RendererShaders::sLineShader = nullptr;
GizmoShader *RendererShaders::sGizmoShader = nullptr;
GridShader *RendererShaders::sGridShader = nullptr;

void RendererShaders::createInternalShaders()
{
    // Note these pointers never free'd but they are static and
    // exist for the length of the program so ... meh?
    Log::warn("Start compling internal shaders\n");
    sStandardShader = new StandardShader();
    sSSAOShader = new SSAOShader();
    sGeometryShader = new GeometryShader();
    sDepthShader = new DepthShader();
    sDepthCubemapShader = new DepthCubemapShader();
    sQuadShader = new QuadShader();
    sSpriteShader = new SpriteShader();
    sGBufferShader = new GBufferShader();
    sColorShader = new ColorShader();
    sColorInstancedShader = new ColorInstancedShader();
    sNormalShader = new NormalShader();
    sNormalInstancedShader = new NormalInstancedShader();
    sPositionShader = new PositionShader();
    sPositionInstancedShader = new PositionInstancedShader();
    sLinearDepthShader = new LinearDepthShader();
    sLinearDepthInstancedShader = new LinearDepthInstancedShader();
    sLineShader = new LineShader();
    sGizmoShader = new GizmoShader();
    sGridShader = new GridShader();
    Log::warn("Finished compiling internal shaders\n");
}

StandardShader *RendererShaders::getStandardShader()
{
    return RendererShaders::sStandardShader;
}

SSAOShader *RendererShaders::getSSAOShader()
{
    return RendererShaders::sSSAOShader;
}

GeometryShader *RendererShaders::getGeometryShader()
{
    return RendererShaders::sGeometryShader;
}

DepthShader *RendererShaders::getDepthShader()
{
    return RendererShaders::sDepthShader;
}

DepthCubemapShader *RendererShaders::getDepthCubemapShader()
{
    return RendererShaders::sDepthCubemapShader;
}

QuadShader *RendererShaders::getScreenQuadShader()
{
    return RendererShaders::sQuadShader;
}

SpriteShader *RendererShaders::getSpriteShader()
{
    return RendererShaders::sSpriteShader;
}

GBufferShader *RendererShaders::getGBufferShader()
{
    return RendererShaders::sGBufferShader;
}

ColorShader *RendererShaders::getColorShader()
{
    return RendererShaders::sColorShader;
}

ColorInstancedShader *RendererShaders::getColorInstancedShader()
{
    return RendererShaders::sColorInstancedShader;
}

NormalShader *RendererShaders::getNormalShader()
{
    return RendererShaders::sNormalShader;
}

NormalInstancedShader *RendererShaders::getNormalInstancedShader()
{
    return RendererShaders::sNormalInstancedShader;
}

PositionShader *RendererShaders::getPositionShader()
{
    return RendererShaders::sPositionShader;
}

PositionInstancedShader *RendererShaders::getPositionInstancedShader()
{
    return RendererShaders::sPositionInstancedShader;
}

LinearDepthShader *RendererShaders::getLinearDepthShader()
{
    return RendererShaders::sLinearDepthShader;
}

LinearDepthInstancedShader *RendererShaders::getLinearDepthInstancedShader()
{
    return RendererShaders::sLinearDepthInstancedShader;
}

LineShader *RendererShaders::getLineShader()
{
    return RendererShaders::sLineShader;
}

GizmoShader *RendererShaders::getGizmoShader()
{
    return RendererShaders::sGizmoShader;
}

GridShader *RendererShaders::getGridShader()
{
    return RendererShaders::sGridShader;
}