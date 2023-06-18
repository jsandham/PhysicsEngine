#ifndef RENDERER_SHADERS_H__
#define RENDERER_SHADERS_H__

#include <string>
#include "ShaderProgram.h"

namespace PhysicsEngine
{
class StandardShader
{
  private:
    ShaderProgram *mShader;

  public:
    StandardShader();
    ~StandardShader();

    void bind();
    void unbind();

    std::string getVertexShader() const;
    std::string getFragmentShader() const;
};

class GBufferShader
{
  private:
    ShaderProgram *mShader;
    int mModelLoc;

  public:
    GBufferShader();
    ~GBufferShader();

    void bind();
    void unbind();

    void setModel(const glm::mat4 &model);
};

class QuadShader
{
  private:
    ShaderProgram *mShader;
    int mScreenTexLoc;

  public:
    QuadShader();
    ~QuadShader();

    void bind();
    void unbind();
    void setScreenTexture(int texUnit, void *tex);
};

class DepthShader
{
  private:
    ShaderProgram *mShader;
    int mModelLoc;
    int mViewLoc;
    int mProjectionLoc;

  public:
    DepthShader();
    ~DepthShader();

    void bind();
    void unbind();
    void setModel(const glm::mat4 &model);
    void setView(const glm::mat4 &view);
    void setProjection(const glm::mat4 &projection);
};

class DepthCubemapShader
{
  private:
    ShaderProgram *mShader;
    int mLightPosLoc;
    int mFarPlaneLoc;
    int mModelLoc;
    int mCubeViewProjMatricesLoc[6];

  public:
    DepthCubemapShader();
    ~DepthCubemapShader();

    void bind();
    void unbind();
    void setLightPos(const glm::vec3 &lightPos);
    void setFarPlane(float farPlane);
    void setModel(const glm::mat4 &model);
    void setCubeViewProj(int index, const glm::mat4 &modelView);
};

class GeometryShader
{
  private:
    ShaderProgram *mShader;
    int mModelLoc;

  public:
    GeometryShader();
    ~GeometryShader();

    void bind();
    void unbind();
    void setModel(const glm::mat4 &model);
};

class NormalShader
{
  private:
    ShaderProgram *mShader;
    int mModelLoc;

  public:
    NormalShader();
    ~NormalShader();

    void bind();
    void unbind();
    void setModel(const glm::mat4 &model);
};

class NormalInstancedShader
{
  private:
    ShaderProgram *mShader;

  public:
    NormalInstancedShader();
    ~NormalInstancedShader();

    void bind();
    void unbind();
};

class PositionShader
{
  private:
    ShaderProgram *mShader;
    int mModelLoc;

  public:
    PositionShader();
    ~PositionShader();

    void bind();
    void unbind();
    void setModel(const glm::mat4 &model);
};

class PositionInstancedShader
{
  private:
    ShaderProgram *mShader;

  public:
    PositionInstancedShader();
    ~PositionInstancedShader();

    void bind();
    void unbind();
};

class LinearDepthShader
{
  private:
    ShaderProgram *mShader;
    int mModelLoc;

  public:
    LinearDepthShader();
    ~LinearDepthShader();

    void bind();
    void unbind();
    void setModel(const glm::mat4 &model);
};

class LinearDepthInstancedShader
{
  private:
    ShaderProgram *mShader;

  public:
    LinearDepthInstancedShader();
    ~LinearDepthInstancedShader();

    void bind();
    void unbind();
};

class ColorShader
{
  private:
    ShaderProgram *mShader;
    int mModelLoc;
    int mColorLoc;

  public:
    ColorShader();
    ~ColorShader();

    void bind();
    void unbind();
    void setModel(const glm::mat4 &model);
    void setColor32(const Color32 &color);
};

class ColorInstancedShader
{
  private:
    ShaderProgram *mShader;

  public:
    ColorInstancedShader();
    ~ColorInstancedShader();

    void bind();
    void unbind();
};

class SSAOShader
{
  private:
    ShaderProgram *mShader;
    int mProjectionLoc;
    int mPositionTexLoc;
    int mNormalTexLoc;
    int mNoiseTexLoc;
    int mSamplesLoc[64];

  public:
    SSAOShader();
    ~SSAOShader();

    void bind();
    void unbind();
    void setProjection(const glm::mat4 &projection);
    void setPositionTexture(int texUnit, void *tex);
    void setNormalTexture(int texUnit, void *tex);
    void setNoiseTexture(int texUnit, void *tex);
    void setSample(int index, const glm::vec3 &sample);
};

class SpriteShader
{
  private:
    ShaderProgram *mShader;
    int mModelLoc;
    int mViewLoc;
    int mProjectionLoc;
    int mColorLoc;
    int mImageLoc;

  public:
    SpriteShader();
    ~SpriteShader();

    void bind();
    void unbind();
    void setModel(const glm::mat4 &model);
    void setView(const glm::mat4 &view);
    void setProjection(const glm::mat4 &projection);
    void setColor(const Color &color);
    void setImage(int texUnit, TextureHandle *tex);
};

class LineShader
{
  private:
    ShaderProgram *mShader;
    int mMVPLoc;

  public:
    LineShader();
    ~LineShader();

    void bind();
    void unbind();
    void setMVP(const glm::mat4 &mvp);
};

class GizmoShader
{
  private:
    ShaderProgram *mShader;
    int mModelLoc;
    int mViewLoc;
    int mProjectionLoc;
    int mColorLoc;
    int mLightPosLoc;

  public:
    GizmoShader();
    ~GizmoShader();

    void bind();
    void unbind();
    void setModel(const glm::mat4 &model);
    void setView(const glm::mat4 &view);
    void setProjection(const glm::mat4 &projection);
    void setColor(const Color &color);
    void setLightPos(const glm::vec3 &lightPos);
};

class GridShader
{
  private:
    ShaderProgram *mShader;
    int mMVPLoc;
    int mColorLoc;

  public:
    GridShader();
    ~GridShader();

    void bind();
    void unbind();
    void setMVP(const glm::mat4 &mvp);
    void setColor(const Color &color);
};


class RendererShaders
{
  private:
    static StandardShader *sStandardShader;
    static SSAOShader *sSSAOShader;
    static GeometryShader *sGeometryShader;
    static DepthShader *sDepthShader;
    static DepthCubemapShader *sDepthCubemapShader;
    static QuadShader *sQuadShader;
    static SpriteShader *sSpriteShader;
    static GBufferShader *sGBufferShader;
    static ColorShader *sColorShader;
    static ColorInstancedShader *sColorInstancedShader;
    static NormalShader *sNormalShader;
    static NormalInstancedShader *sNormalInstancedShader;
    static PositionShader *sPositionShader;
    static PositionInstancedShader *sPositionInstancedShader;
    static LinearDepthShader *sLinearDepthShader;
    static LinearDepthInstancedShader *sLinearDepthInstancedShader;
    static LineShader *sLineShader;
    static GizmoShader *sGizmoShader;
    static GridShader *sGridShader;

  public:
    static StandardShader *getStandardShader();
    static SSAOShader *getSSAOShader();
    static GeometryShader *getGeometryShader();
    static DepthShader *getDepthShader();
    static DepthCubemapShader *getDepthCubemapShader();
    static QuadShader *getScreenQuadShader();
    static SpriteShader *getSpriteShader();
    static GBufferShader *getGBufferShader();
    static ColorShader *getColorShader();
    static ColorInstancedShader *getColorInstancedShader();
    static NormalShader *getNormalShader();
    static NormalInstancedShader *getNormalInstancedShader();
    static PositionShader *getPositionShader();
    static PositionInstancedShader *getPositionInstancedShader();
    static LinearDepthShader *getLinearDepthShader();
    static LinearDepthInstancedShader *getLinearDepthInstancedShader();
    static LineShader *getLineShader();
    static GizmoShader *getGizmoShader();
    static GridShader *getGridShader();

    static void createInternalShaders();
};
} // namespace PhysicsEngine

#endif