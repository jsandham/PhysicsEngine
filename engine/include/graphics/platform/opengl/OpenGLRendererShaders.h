#ifndef OPENGL_RENDERER_SHADERS_H__
#define OPENGL_RENDERER_SHADERS_H__

#include "../../RendererShaders.h"

namespace PhysicsEngine
{
class OpenGLStandardShader : public StandardShader
{
  private:
    ShaderProgram *mShader;

  public:
    OpenGLStandardShader();
    ~OpenGLStandardShader();

    void bind() override;
    void unbind() override;

    std::string getVertexShader() override;
    std::string getFragmentShader() override;
};

class OpenGLGBufferShader : public GBufferShader
{
  private:
    ShaderProgram *mShader;
    int mModelLoc;

  public:
    OpenGLGBufferShader();
    ~OpenGLGBufferShader();

    void bind() override;
    void unbind() override;

    void setModel(const glm::mat4 &model) override;
};

class OpenGLQuadShader : public QuadShader
{
  private:
    ShaderProgram *mShader;
    int mScreenTexLoc;

  public:
    OpenGLQuadShader();
    ~OpenGLQuadShader();

    void bind() override;
    void unbind() override;
    void setScreenTexture(int texUnit, TextureHandle *tex) override;
};

class OpenGLDepthShader : public DepthShader
{
  private:
    ShaderProgram *mShader;
    int mModelLoc;
    int mViewLoc;
    int mProjectionLoc;

  public:
    OpenGLDepthShader();
    ~OpenGLDepthShader();

    void bind() override;
    void unbind() override;
    void setModel(const glm::mat4 &model) override;
    void setView(const glm::mat4 &view) override;
    void setProjection(const glm::mat4 &projection) override;
};

class OpenGLDepthCubemapShader : public DepthCubemapShader
{
  private:
    ShaderProgram *mShader;
    int mLightPosLoc;
    int mFarPlaneLoc;
    int mModelLoc;
    int mCubeViewProjMatricesLoc[6];

  public:
    OpenGLDepthCubemapShader();
    ~OpenGLDepthCubemapShader();

    void bind() override;
    void unbind() override;
    void setLightPos(const glm::vec3 &lightPos) override;
    void setFarPlane(float farPlane) override;
    void setModel(const glm::mat4 &model) override;
    void setCubeViewProj(int index, const glm::mat4 &modelView) override;
};

class OpenGLGeometryShader : public GeometryShader
{
  private:
    ShaderProgram *mShader;
    int mModelLoc;

  public:
    OpenGLGeometryShader();
    ~OpenGLGeometryShader();

    void bind() override;
    void unbind() override;
    void setModel(const glm::mat4 &model) override;
};

class OpenGLNormalShader : public NormalShader
{
  private:
    ShaderProgram *mShader;
    int mModelLoc;

  public:
    OpenGLNormalShader();
    ~OpenGLNormalShader();

    void bind() override;
    void unbind() override;
    void setModel(const glm::mat4 &model) override;
};

class OpenGLNormalInstancedShader : public NormalInstancedShader
{
  private:
    ShaderProgram *mShader;

  public:
    OpenGLNormalInstancedShader();
    ~OpenGLNormalInstancedShader();

    void bind() override;
    void unbind() override;
};

class OpenGLPositionShader : public PositionShader
{
  private:
    ShaderProgram *mShader;
    int mModelLoc;

  public:
    OpenGLPositionShader();
    ~OpenGLPositionShader();

    void bind() override;
    void unbind() override;
    void setModel(const glm::mat4 &model) override;
};

class OpenGLPositionInstancedShader : public PositionInstancedShader
{
  private:
    ShaderProgram *mShader;

  public:
    OpenGLPositionInstancedShader();
    ~OpenGLPositionInstancedShader();

    void bind() override;
    void unbind() override;
};

class OpenGLLinearDepthShader : public LinearDepthShader
{
  private:
    ShaderProgram *mShader;
    int mModelLoc;

  public:
    OpenGLLinearDepthShader();
    ~OpenGLLinearDepthShader();

    void bind() override;
    void unbind() override;
    void setModel(const glm::mat4 &model) override;
};

class OpenGLLinearDepthInstancedShader : public LinearDepthInstancedShader
{
  private:
    ShaderProgram *mShader;

  public:
    OpenGLLinearDepthInstancedShader();
    ~OpenGLLinearDepthInstancedShader();

    void bind() override;
    void unbind() override;
};

class OpenGLColorShader : public ColorShader
{
  private:
    ShaderProgram *mShader;
    int mModelLoc;
    int mColorLoc;

  public:
    OpenGLColorShader();
    ~OpenGLColorShader();

    void bind() override;
    void unbind() override;
    void setModel(const glm::mat4 &model) override;
    void setColor32(const Color32 &color) override;
};

class OpenGLColorInstancedShader : public ColorInstancedShader
{
  private:
    ShaderProgram *mShader;

  public:
    OpenGLColorInstancedShader();
    ~OpenGLColorInstancedShader();

    void bind() override;
    void unbind() override;
};

class OpenGLSSAOShader : public SSAOShader
{
  private:
    ShaderProgram *mShader;
    int mProjectionLoc;
    int mPositionTexLoc;
    int mNormalTexLoc;
    int mNoiseTexLoc;
    int mSamplesLoc[64];

  public:
    OpenGLSSAOShader();
    ~OpenGLSSAOShader();

    void bind() override;
    void unbind() override;
    void setProjection(const glm::mat4 &projection) override;
    void setPositionTexture(int texUnit, TextureHandle *tex) override;
    void setNormalTexture(int texUnit, TextureHandle *tex) override;
    void setNoiseTexture(int texUnit, TextureHandle *tex) override;
    void setSample(int index, const glm::vec3 &sample) override;
};

class OpenGLSpriteShader : public SpriteShader
{
  private:
    ShaderProgram *mShader;
    int mModelLoc;
    int mViewLoc;
    int mProjectionLoc;
    int mColorLoc;
    int mImageLoc;

  public:
    OpenGLSpriteShader();
    ~OpenGLSpriteShader();

    void bind() override;
    void unbind() override;
    void setModel(const glm::mat4 &model) override;
    void setView(const glm::mat4 &view) override;
    void setProjection(const glm::mat4 &projection) override;
    void setColor(const Color &color) override;
    void setImage(int texUnit, TextureHandle *tex) override;
};

class OpenGLLineShader : public LineShader
{
  private:
    ShaderProgram *mShader;
    int mMVPLoc;

  public:
    OpenGLLineShader();
    ~OpenGLLineShader();

    void bind() override;
    void unbind() override;
    void setMVP(const glm::mat4 &mvp) override;
};

class OpenGLGizmoShader : public GizmoShader
{
  private:
    ShaderProgram *mShader;
    int mModelLoc;
    int mViewLoc;
    int mProjectionLoc;
    int mColorLoc;
    int mLightPosLoc;

  public:
    OpenGLGizmoShader();
    ~OpenGLGizmoShader();

    void bind() override;
    void unbind() override;
    void setModel(const glm::mat4 &model) override;
    void setView(const glm::mat4 &view) override;
    void setProjection(const glm::mat4 &projection) override;
    void setColor(const Color &color) override;
    void setLightPos(const glm::vec3 &lightPos) override;
};

class OpenGLGridShader : public GridShader
{
  private:
    ShaderProgram *mShader;
    int mMVPLoc;
    int mColorLoc;

  public:
    OpenGLGridShader();
    ~OpenGLGridShader();

    void bind() override;
    void unbind() override;
    void setMVP(const glm::mat4 &mvp) override;
    void setColor(const Color &color) override;
};
} // namespace PhysicsEngine

#endif