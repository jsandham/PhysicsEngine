#include "../../../../include/graphics/platform/opengl/OpenGLRendererShaders.h"
#include "../../../../include/graphics/platform/opengl/OpenGLRenderer.h"
#include "../../../../include/core/Shader.h"
#include "../../../../include/core/Log.h"

#include "GLSL/glsl_shaders.h"

using namespace PhysicsEngine;

OpenGLStandardShader::OpenGLStandardShader()
{
    mShader = ShaderProgram::create();
    mShader->load("Standard", getStandardVertexShader(), getStandardFragmentShader());
    mShader->compile();
}

OpenGLStandardShader::~OpenGLStandardShader()
{
    delete mShader;
}

void OpenGLStandardShader::bind()
{
    mShader->bind();
}

void OpenGLStandardShader::unbind()
{
    mShader->unbind();
}

std::string OpenGLStandardShader::getVertexShader()
{
    return getStandardVertexShader();
}

std::string OpenGLStandardShader::getFragmentShader()
{
    return getStandardFragmentShader();
}

OpenGLGBufferShader::OpenGLGBufferShader()
{
    mShader = ShaderProgram::create();
    mShader->load("GBuffer", getGBufferVertexShader(), getGBufferFragmentShader());
    mShader->compile();

    mModelLoc = mShader->findUniformLocation("model");
}

OpenGLGBufferShader ::~OpenGLGBufferShader()
{
    delete mShader;
}

void OpenGLGBufferShader::bind()
{
    mShader->bind();
}

void OpenGLGBufferShader::unbind()
{
    mShader->unbind();
}

void OpenGLGBufferShader::setModel(const glm::mat4 &model)
{
    mShader->setMat4(mModelLoc, model);
}

OpenGLQuadShader::OpenGLQuadShader()
{
    mShader = ShaderProgram::create();
    mShader->load("ScreenQuad", getScreenQuadVertexShader(), getScreenQuadFragmentShader());
    mShader->compile();

    mScreenTexLoc = mShader->findUniformLocation("screenTexture");
}

OpenGLQuadShader ::~OpenGLQuadShader()
{
    delete mShader;
}

void OpenGLQuadShader::bind()
{
    mShader->bind();
}

void OpenGLQuadShader::unbind()
{
    mShader->unbind();
}

void OpenGLQuadShader::setScreenTexture(int texUnit, TextureHandle *tex)
{
    mShader->setTexture2D(mScreenTexLoc, texUnit, tex);
}

OpenGLDepthShader::OpenGLDepthShader()
{
    mShader = ShaderProgram::create();
    mShader->load("DepthMap", getShadowDepthMapVertexShader(), getShadowDepthMapFragmentShader());
    mShader->compile();

    mModelLoc = mShader->findUniformLocation("model");
    mViewLoc = mShader->findUniformLocation("view");
    mProjectionLoc = mShader->findUniformLocation("projection");
}

OpenGLDepthShader ::~OpenGLDepthShader()
{
    delete mShader;
}

void OpenGLDepthShader::bind()
{
    mShader->bind();
}

void OpenGLDepthShader::unbind()
{
    mShader->unbind();
}

void OpenGLDepthShader::setModel(const glm::mat4 &model)
{
    mShader->setMat4(mModelLoc, model);
}

void OpenGLDepthShader::setView(const glm::mat4 &view)
{
    mShader->setMat4(mViewLoc, view);
}

void OpenGLDepthShader::setProjection(const glm::mat4 &projection)
{
    mShader->setMat4(mProjectionLoc, projection);
}

OpenGLDepthCubemapShader::OpenGLDepthCubemapShader()
{
    mShader = ShaderProgram::create();
    mShader->load("DepthCubemap", getShadowDepthCubemapVertexShader(), getShadowDepthCubemapFragmentShader(),
                  getShadowDepthCubemapGeometryShader());
    mShader->compile();

    mLightPosLoc = mShader->findUniformLocation("lightPos");
    mFarPlaneLoc = mShader->findUniformLocation("farPlane");
    mModelLoc = mShader->findUniformLocation("model");
    mCubeViewProjMatricesLoc[0] = mShader->findUniformLocation("cubeViewProjMatrices[0]");
    mCubeViewProjMatricesLoc[1] = mShader->findUniformLocation("cubeViewProjMatrices[1]");
    mCubeViewProjMatricesLoc[2] = mShader->findUniformLocation("cubeViewProjMatrices[2]");
    mCubeViewProjMatricesLoc[3] = mShader->findUniformLocation("cubeViewProjMatrices[3]");
    mCubeViewProjMatricesLoc[4] = mShader->findUniformLocation("cubeViewProjMatrices[4]");
    mCubeViewProjMatricesLoc[5] = mShader->findUniformLocation("cubeViewProjMatrices[5]");
}

OpenGLDepthCubemapShader::~OpenGLDepthCubemapShader()
{
    delete mShader;
}

void OpenGLDepthCubemapShader::bind()
{
    mShader->bind();
}

void OpenGLDepthCubemapShader::unbind()
{
    mShader->unbind();
}

void OpenGLDepthCubemapShader::setLightPos(const glm::vec3 &lightPos)
{
    mShader->setVec3(mLightPosLoc, lightPos);
}

void OpenGLDepthCubemapShader::setFarPlane(float farPlane)
{
    mShader->setFloat(mFarPlaneLoc, farPlane);
}

void OpenGLDepthCubemapShader::setModel(const glm::mat4 &model)
{
    mShader->setMat4(mModelLoc, model);
}

void OpenGLDepthCubemapShader::setCubeViewProj(int index, const glm::mat4 &modelView)
{
    assert(index >= 0);
    assert(index <= 5);
    mShader->setMat4(mCubeViewProjMatricesLoc[index], modelView);
}

OpenGLGeometryShader::OpenGLGeometryShader()
{
    mShader = ShaderProgram::create();
    mShader->load("Geometry", getGeometryVertexShader(), getGeometryFragmentShader());
    mShader->compile();

    mModelLoc = mShader->findUniformLocation("model");
}

OpenGLGeometryShader::~OpenGLGeometryShader()
{
    delete mShader;
}

void OpenGLGeometryShader::bind()
{
    mShader->bind();
}

void OpenGLGeometryShader::unbind()
{
    mShader->unbind();
}

void OpenGLGeometryShader::setModel(const glm::mat4 &model)
{
    mShader->setMat4(mModelLoc, model);
}

OpenGLNormalShader::OpenGLNormalShader()
{
    mShader = ShaderProgram::create();
    mShader->load("Normal", getNormalVertexShader(), getNormalFragmentShader());
    mShader->compile();

    mModelLoc = mShader->findUniformLocation("model");
}

OpenGLNormalShader::~OpenGLNormalShader()
{
    delete mShader;
}

void OpenGLNormalShader::bind()
{
    mShader->bind();
}

void OpenGLNormalShader::unbind()
{
    mShader->unbind();
}

void OpenGLNormalShader::setModel(const glm::mat4 &model)
{
    mShader->setMat4(mModelLoc, model);
}

OpenGLNormalInstancedShader::OpenGLNormalInstancedShader()
{
    mShader = ShaderProgram::create();
    mShader->load("Normal Instanced", getNormalInstancedVertexShader(), getNormalInstancedFragmentShader());
    mShader->compile();
}

OpenGLNormalInstancedShader::~OpenGLNormalInstancedShader()
{
    delete mShader;
}

void OpenGLNormalInstancedShader::bind()
{
    mShader->bind();
}

void OpenGLNormalInstancedShader::unbind()
{
    mShader->unbind();
}

OpenGLPositionShader::OpenGLPositionShader()
{
    mShader = ShaderProgram::create();
    mShader->load("Position", getPositionVertexShader(), getPositionFragmentShader());
    mShader->compile();

    mModelLoc = mShader->findUniformLocation("model");
}

OpenGLPositionShader::~OpenGLPositionShader()
{
    delete mShader;
}

void OpenGLPositionShader::bind()
{
    mShader->bind();
}

void OpenGLPositionShader::unbind()
{
    mShader->unbind();
}

void OpenGLPositionShader::setModel(const glm::mat4 &model)
{
    mShader->setMat4(mModelLoc, model);
}

OpenGLPositionInstancedShader::OpenGLPositionInstancedShader()
{
    mShader = ShaderProgram::create();
    mShader->load("Position Instanced", getPositionInstancedVertexShader(), getPositionInstancedFragmentShader());
    mShader->compile();
}

OpenGLPositionInstancedShader::~OpenGLPositionInstancedShader()
{
    delete mShader;
}

void OpenGLPositionInstancedShader::bind()
{
    mShader->bind();
}

void OpenGLPositionInstancedShader::unbind()
{
    mShader->unbind();
}

OpenGLLinearDepthShader::OpenGLLinearDepthShader()
{
    mShader = ShaderProgram::create();
    mShader->load("LinearDepth", getLinearDepthVertexShader(), getLinearDepthFragmentShader());
    mShader->compile();

    mModelLoc = mShader->findUniformLocation("model");
}

OpenGLLinearDepthShader::~OpenGLLinearDepthShader()
{
    delete mShader;
}

void OpenGLLinearDepthShader::bind()
{
    mShader->bind();
}

void OpenGLLinearDepthShader::unbind()
{
    mShader->unbind();
}

void OpenGLLinearDepthShader::setModel(const glm::mat4 &model)
{
    mShader->setMat4(mModelLoc, model);
}

OpenGLLinearDepthInstancedShader::OpenGLLinearDepthInstancedShader()
{
    mShader = ShaderProgram::create();
    mShader->load("Linear Depth Instanced", getLinearDepthInstancedVertexShader(), getLinearDepthInstancedFragmentShader());
    mShader->compile();
}

OpenGLLinearDepthInstancedShader::~OpenGLLinearDepthInstancedShader()
{
    delete mShader;
}

void OpenGLLinearDepthInstancedShader::bind()
{
    mShader->bind();
}

void OpenGLLinearDepthInstancedShader::unbind()
{
    mShader->unbind();
}

OpenGLColorShader::OpenGLColorShader()
{
    mShader = ShaderProgram::create();
    mShader->load("Color", getColorVertexShader(), getColorFragmentShader());
    mShader->compile();

    mModelLoc = mShader->findUniformLocation("model");
    mColorLoc = mShader->findUniformLocation("material.color");
}

OpenGLColorShader::~OpenGLColorShader()
{
    delete mShader;
}

void OpenGLColorShader::bind()
{
    mShader->bind();
}

void OpenGLColorShader::unbind()
{
    mShader->unbind();
}

void OpenGLColorShader::setModel(const glm::mat4 &model)
{
    mShader->setMat4(mModelLoc, model);
}

void OpenGLColorShader::setColor32(const Color32 &color)
{
    mShader->setColor32(mColorLoc, color);
}

OpenGLColorInstancedShader::OpenGLColorInstancedShader()
{
    mShader = ShaderProgram::create();
    mShader->load("Color Instanced", getColorInstancedVertexShader(),
                  getColorInstancedFragmentShader());
    mShader->compile();
}

OpenGLColorInstancedShader::~OpenGLColorInstancedShader()
{
    delete mShader;
}

void OpenGLColorInstancedShader::bind()
{
    mShader->bind();
}

void OpenGLColorInstancedShader::unbind()
{
    mShader->unbind();
}

OpenGLSSAOShader::OpenGLSSAOShader()
{
    mShader = ShaderProgram::create();
    mShader->load("SSAO", getSSAOVertexShader(), getSSAOFragmentShader());
    mShader->compile();

    mProjectionLoc = mShader->findUniformLocation("projection");
    mPositionTexLoc = mShader->findUniformLocation("positionTex");
    mNormalTexLoc = mShader->findUniformLocation("normalTex");
    mNoiseTexLoc = mShader->findUniformLocation("noiseTex");
    for (int i = 0; i < 64; i++)
    {
        mSamplesLoc[i] = mShader->findUniformLocation("samples[" + std::to_string(i) + "]");
    }
}
    
OpenGLSSAOShader::~OpenGLSSAOShader()
{
    delete mShader;
}

void OpenGLSSAOShader::bind()
{
    mShader->bind();
}

void OpenGLSSAOShader::unbind()
{
    mShader->unbind();
}

void OpenGLSSAOShader::setProjection(const glm::mat4 &projection)
{
    mShader->setMat4(mProjectionLoc, projection);
}

void OpenGLSSAOShader::setPositionTexture(int texUnit, TextureHandle *tex)
{
    mShader->setTexture2D(mNormalTexLoc, texUnit, tex);
}

void OpenGLSSAOShader::setNormalTexture(int texUnit, TextureHandle *tex)
{
    mShader->setTexture2D(mPositionTexLoc, texUnit, tex);
}

void OpenGLSSAOShader::setNoiseTexture(int texUnit, TextureHandle *tex)
{
    mShader->setTexture2D(mNoiseTexLoc, texUnit, tex);
}

void OpenGLSSAOShader::setSample(int index, const glm::vec3 &sample)
{
    assert(index >= 0);
    assert(index <= 63);
    mShader->setVec3(mSamplesLoc[index], sample);
}

OpenGLSpriteShader::OpenGLSpriteShader()
{
    mShader = ShaderProgram::create();
    mShader->load("Sprite", getSpriteVertexShader(), getSpriteFragmentShader());
    mShader->compile();

    mModelLoc = mShader->findUniformLocation("model");
    mViewLoc = mShader->findUniformLocation("view");
    mProjectionLoc = mShader->findUniformLocation("projection");
    mColorLoc = mShader->findUniformLocation("spriteColor");
    mImageLoc = mShader->findUniformLocation("image");
}

OpenGLSpriteShader::~OpenGLSpriteShader()
{
    delete mShader;
}

void OpenGLSpriteShader::bind()
{
    mShader->bind();
}

void OpenGLSpriteShader::unbind()
{
    mShader->unbind();
}

void OpenGLSpriteShader::setModel(const glm::mat4 &model)
{
    mShader->setMat4(mModelLoc, model);
}
   
void OpenGLSpriteShader::setView(const glm::mat4 &view)
{
    mShader->setMat4(mViewLoc, view);
}

void OpenGLSpriteShader::setProjection(const glm::mat4 &projection)
{
    mShader->setMat4(mProjectionLoc, projection);
}
  
void OpenGLSpriteShader::setColor(const Color &color)
{
    mShader->setColor(mColorLoc, color);
}

void OpenGLSpriteShader::setImage(int texUnit, TextureHandle *tex)
{
    mShader->setTexture2D(mImageLoc, texUnit, tex);
}

OpenGLLineShader::OpenGLLineShader()
{
    mShader = ShaderProgram::create();
    mShader->load("Line", getLineVertexShader(), getLineFragmentShader());
    mShader->compile();

    mMVPLoc = mShader->findUniformLocation("mvp");
}

OpenGLLineShader::~OpenGLLineShader()
{
    delete mShader;
}

void OpenGLLineShader::bind()
{
    mShader->bind();
}

void OpenGLLineShader::unbind()
{
    mShader->unbind();
}

void OpenGLLineShader::setMVP(const glm::mat4 &mvp)
{
    mShader->setMat4(mMVPLoc, mvp);
}

OpenGLGizmoShader::OpenGLGizmoShader()
{
    mShader = ShaderProgram::create();
    mShader->load("Gizmo", getGizmoVertexShader(), getGizmoFragmentShader());
    mShader->compile();

    mModelLoc = mShader->findUniformLocation("model");
    mViewLoc = mShader->findUniformLocation("view");
    mProjectionLoc = mShader->findUniformLocation("projection");
    mColorLoc = mShader->findUniformLocation("color");
    mLightPosLoc = mShader->findUniformLocation("lightPos");
}

OpenGLGizmoShader::~OpenGLGizmoShader()
{
    delete mShader;
}

void OpenGLGizmoShader::bind()
{
    mShader->bind();
}

void OpenGLGizmoShader::unbind()
{
    mShader->unbind();
}

void OpenGLGizmoShader::setModel(const glm::mat4 &model)
{
    mShader->setMat4(mModelLoc, model);
}

void OpenGLGizmoShader::setView(const glm::mat4 &view)
{
    mShader->setMat4(mViewLoc, view);
}

void OpenGLGizmoShader::setProjection(const glm::mat4 &projection)
{
    mShader->setMat4(mProjectionLoc, projection);
}

void OpenGLGizmoShader::setColor(const Color &color)
{
    mShader->setColor(mColorLoc, color);
}

void OpenGLGizmoShader::setLightPos(const glm::vec3 &lightPos)
{
    mShader->setVec3(mLightPosLoc, lightPos);
}

OpenGLGridShader::OpenGLGridShader()
{
    mShader = ShaderProgram::create();
    mShader->load("Grid", getGridVertexShader(), getGridFragmentShader());
    mShader->compile();

    mMVPLoc = mShader->findUniformLocation("mvp");
    mColorLoc = mShader->findUniformLocation("color");
}

OpenGLGridShader::~OpenGLGridShader()
{
    delete mShader;
}

void OpenGLGridShader::bind()
{
    mShader->bind();
}

void OpenGLGridShader::unbind()
{
    mShader->unbind();
}

void OpenGLGridShader::setMVP(const glm::mat4 &mvp)
{
    mShader->setMat4(mMVPLoc, mvp);
}

void OpenGLGridShader::setColor(const Color &color)
{
    mShader->setColor(mColorLoc, color);
}