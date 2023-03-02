#ifndef DIRECTX_RENDERER_SHADERS_H__
#define DIRECTX_RENDERER_SHADERS_H__

#include "../../RendererShaders.h"

namespace PhysicsEngine
{
    class DirectXStandardShader : public StandardShader
    {
      public:
        DirectXStandardShader();
        ~DirectXStandardShader();

        void bind() override;
        void unbind() override;

        std::string getVertexShader() override;
        std::string getFragmentShader() override;
    };

    class DirectXGBufferShader : public GBufferShader
    {
      public:
        DirectXGBufferShader();
        ~DirectXGBufferShader();

        void bind() override;
        void unbind() override;
        void setModel(const glm::mat4 &model) override;
    };

    class DirectXQuadShader : public QuadShader
    {
      public:
        DirectXQuadShader();
        ~DirectXQuadShader();

        void bind() override;
        void unbind() override;
        void setScreenTexture(int texUnit, TextureHandle *tex) override;
    };

    class DirectXDepthShader : public DepthShader
    {
      public:
        DirectXDepthShader();
        ~DirectXDepthShader();

        void bind() override;
        void unbind() override;
        void setModel(const glm::mat4 &model) override;
        void setView(const glm::mat4 &view) override;
        void setProjection(const glm::mat4 &projection) override;
    };

    class DirectXDepthCubemapShader : public DepthCubemapShader
    {
      public:
        DirectXDepthCubemapShader();
        ~DirectXDepthCubemapShader();

        void bind() override;
        void unbind() override;
        void setLightPos(const glm::vec3 &lightPos) override;
        void setFarPlane(float farPlane) override;
        void setModel(const glm::mat4 &model) override;
        void setCubeViewProj(int index, const glm::mat4 &modelView) override;
    };

    class DirectXGeometryShader : public GeometryShader
    {
      public:
        DirectXGeometryShader();
        ~DirectXGeometryShader();

        void bind() override;
        void unbind() override;
        void setModel(const glm::mat4 &model) override;
    };

    class DirectXNormalShader : public NormalShader
    {
      public:
        DirectXNormalShader();
        ~DirectXNormalShader();

        void bind() override;
        void unbind() override;
        void setModel(const glm::mat4 &model) override;
    };

    class DirectXNormalInstancedShader : public NormalInstancedShader
    {
      private:
        ShaderProgram *mShader;

      public:
        DirectXNormalInstancedShader();
        ~DirectXNormalInstancedShader();

        void bind() override;
        void unbind() override;
    };

    class DirectXPositionShader : public PositionShader
    {
      public:
        DirectXPositionShader();
        ~DirectXPositionShader();

        void bind() override;
        void unbind() override;
        void setModel(const glm::mat4 &model) override;
    };

     class DirectXPositionInstancedShader : public PositionInstancedShader
    {
      private:
        ShaderProgram *mShader;

      public:
        DirectXPositionInstancedShader();
        ~DirectXPositionInstancedShader();

        void bind() override;
        void unbind() override;
    };

    class DirectXLinearDepthShader : public LinearDepthShader
    {
      public:
        DirectXLinearDepthShader();
        ~DirectXLinearDepthShader();
        
        void bind() override;
        void unbind() override;
        void setModel(const glm::mat4 &model) override;
    };

    class DirectXLinearDepthInstancedShader : public LinearDepthInstancedShader
    {
      private:
        ShaderProgram *mShader;

      public:
        DirectXLinearDepthInstancedShader();
        ~DirectXLinearDepthInstancedShader();

        void bind() override;
        void unbind() override;
    };


    class DirectXColorShader : public ColorShader
    {
      public:
        DirectXColorShader();
        ~DirectXColorShader();

        void bind() override;
        void unbind() override;
        void setModel(const glm::mat4 &model) override;
        void setColor32(const Color32 &color) override;
    };

    class DirectXColorInstancedShader : public ColorInstancedShader
    {
      private:
        ShaderProgram *mShader;

      public:
        DirectXColorInstancedShader();
        ~DirectXColorInstancedShader();

        void bind() override;
        void unbind() override;
    };


    class DirectXSSAOShader : public SSAOShader
    {
      public:
        DirectXSSAOShader();
        ~DirectXSSAOShader();

        void bind() override;
        void unbind() override;
        void setProjection(const glm::mat4 &projection) override;
        void setPositionTexture(int texUnit, TextureHandle *tex) override;
        void setNormalTexture(int texUnit, TextureHandle *tex) override;
        void setNoiseTexture(int texUnit, TextureHandle *tex) override;
        void setSample(int index, const glm::vec3 &sample) override;
    };

    class DirectXSpriteShader : public SpriteShader
    {
      public:
        DirectXSpriteShader();
        ~DirectXSpriteShader();

        void bind() override;
        void unbind() override;
        void setModel(const glm::mat4 &model) override;
        void setView(const glm::mat4 &view) override;
        void setProjection(const glm::mat4 &projection) override;
        void setColor(const Color &color) override;
        void setImage(int texUnit, TextureHandle *tex) override;
    };

    class DirectXLineShader : public LineShader
    {
      public:
        DirectXLineShader();
        ~DirectXLineShader();

        void bind() override;
        void unbind() override;
        void setMVP(const glm::mat4 &mvp) override;
    };

    class DirectXGizmoShader : public GizmoShader
    {
      public:
        DirectXGizmoShader();
        ~DirectXGizmoShader();

        void bind() override;
        void unbind() override;
        void setModel(const glm::mat4 &model) override;
        void setView(const glm::mat4 &view) override;
        void setProjection(const glm::mat4 &projection) override;
        void setColor(const Color &color) override;
        void setLightPos(const glm::vec3 &lightPos) override;
    };

    class DirectXGridShader : public GridShader
    {
      public:
        DirectXGridShader();
        ~DirectXGridShader();

        void bind() override;
        void unbind() override;
        void setMVP(const glm::mat4 &mvp) override;
        void setColor(const Color &color) override;
    };
}

#endif
