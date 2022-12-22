#ifndef RENDERER_SHADERS_H__
#define RENDERER_SHADERS_H__

#include <string>
#include "ShaderProgram.h"

namespace PhysicsEngine
{
    class StandardShader
    {
      public:
        StandardShader()
        {
        }
        virtual ~StandardShader(){};

        virtual void bind() = 0;
        virtual void unbind() = 0;

        virtual std::string getVertexShader() = 0;
        virtual std::string getFragmentShader() = 0;

        static StandardShader *create();
    };
    
    class GBufferShader
    {
      public:
        GBufferShader()
        {
        }
        virtual ~GBufferShader(){};

        virtual void bind() = 0;
        virtual void unbind() = 0;

        virtual void setModel(const glm::mat4 &model) = 0;

        static GBufferShader *create();
    };

    class QuadShader
    {
      public:
        QuadShader()
        {
        }
        virtual ~QuadShader(){};

        virtual void bind() = 0;
        virtual void unbind() = 0;
        virtual void setScreenTexture(int texUnit, TextureHandle *tex) = 0;

        static QuadShader *create();
    };

    class DepthShader
    {
      public:
        DepthShader(){};
        virtual ~DepthShader(){};

        virtual void bind() = 0;
        virtual void unbind() = 0;
        virtual void setModel(const glm::mat4 &model) = 0;
        virtual void setView(const glm::mat4 &view) = 0;
        virtual void setProjection(const glm::mat4 &projection) = 0;

        static DepthShader *create();
    };

    class DepthCubemapShader
    {
      public:
        DepthCubemapShader(){};
        virtual ~DepthCubemapShader(){};

        virtual void bind() = 0;
        virtual void unbind() = 0;
        virtual void setLightPos(const glm::vec3 &lightPos) = 0;
        virtual void setFarPlane(float farPlane) = 0;
        virtual void setModel(const glm::mat4 &model) = 0;
        virtual void setCubeViewProj(int index, const glm::mat4 &modelView) = 0;

        static DepthCubemapShader *create();
    };

    class GeometryShader
    {
      public:
        GeometryShader(){};
        virtual ~GeometryShader(){};

        virtual void bind() = 0;
        virtual void unbind() = 0;
        virtual void setModel(const glm::mat4 &model) = 0;

        static GeometryShader *create();
    };

    class NormalShader
    {
      public:
        NormalShader(){};
        virtual ~NormalShader(){};

        virtual void bind() = 0;
        virtual void unbind() = 0;
        virtual void setModel(const glm::mat4 &model) = 0;

        static NormalShader *create();
    };

    class NormalInstancedShader
    {
      public:
        NormalInstancedShader(){};
        virtual ~NormalInstancedShader(){};

        virtual void bind() = 0;
        virtual void unbind() = 0;

        static NormalInstancedShader *create();
    };

    class PositionShader
    {
      public:
        PositionShader(){};
        virtual ~PositionShader(){};

        virtual void bind() = 0;
        virtual void unbind() = 0;
        virtual void setModel(const glm::mat4 &model) = 0;

        static PositionShader *create();
    };

    class PositionInstancedShader
    {
      public:
        PositionInstancedShader(){};
        virtual ~PositionInstancedShader(){};

        virtual void bind() = 0;
        virtual void unbind() = 0;

        static PositionInstancedShader *create();
    };

    class LinearDepthShader
    {
      public:
        LinearDepthShader(){};
        virtual ~LinearDepthShader(){};

        virtual void bind() = 0;
        virtual void unbind() = 0;
        virtual void setModel(const glm::mat4 &model) = 0;

        static LinearDepthShader *create();
    };

    class LinearDepthInstancedShader
    {
      public:
        LinearDepthInstancedShader(){};
        virtual ~LinearDepthInstancedShader(){};

        virtual void bind() = 0;
        virtual void unbind() = 0;

        static LinearDepthInstancedShader *create();
    };

    class ColorShader
    {
      public:
        ColorShader(){};
        virtual ~ColorShader(){};

        virtual void bind() = 0;
        virtual void unbind() = 0;
        virtual void setModel(const glm::mat4 &model) = 0;
        virtual void setColor(const Color32 &color) = 0;

        static ColorShader *create();
    };

    class ColorInstancedShader
    {
      public:
        ColorInstancedShader(){};
        virtual ~ColorInstancedShader(){};

        virtual void bind() = 0;
        virtual void unbind() = 0;

        static ColorInstancedShader *create();
    };

    class SSAOShader
    {
      public:
        SSAOShader(){};
        virtual ~SSAOShader(){};

        virtual void bind() = 0;
        virtual void unbind() = 0;
        virtual void setProjection(const glm::mat4 &projection) = 0;
        virtual void setPositionTexture(int texUnit, TextureHandle *tex) = 0;
        virtual void setNormalTexture(int texUnit, TextureHandle *tex) = 0;
        virtual void setNoiseTexture(int texUnit, TextureHandle *tex) = 0;
        virtual void setSample(int index, const glm::vec3 &sample) = 0;

        static SSAOShader *create();
    };

    class SpriteShader
    {
      public:
        SpriteShader(){};
        virtual ~SpriteShader(){};

        virtual void bind() = 0;
        virtual void unbind() = 0;
        virtual void setModel(const glm::mat4 &model) = 0;
        virtual void setView(const glm::mat4 &view) = 0;
        virtual void setProjection(const glm::mat4 &projection) = 0;
        virtual void setColor(const Color &color) = 0;
        virtual void setImage(int texUnit, TextureHandle *tex) = 0;

        static SpriteShader *create();
    };

    class LineShader
    {
      public:
        LineShader(){};
        virtual ~LineShader(){};

        virtual void bind() = 0;
        virtual void unbind() = 0;
        virtual void setMVP(const glm::mat4 &mvp) = 0;

        static LineShader *create();
    };

    class GizmoShader
    {
      public:
        GizmoShader(){};
        virtual ~GizmoShader(){};

        virtual void bind() = 0;
        virtual void unbind() = 0;
        virtual void setModel(const glm::mat4 &model) = 0;
        virtual void setView(const glm::mat4 &view) = 0;
        virtual void setProjection(const glm::mat4 &projection) = 0;
        virtual void setColor(const Color &color) = 0;
        virtual void setLightPos(const glm::vec3 &lightPos) = 0;

        static GizmoShader *create();
    };

    class GridShader
    {
      public:
        GridShader(){};
        virtual ~GridShader(){};

        virtual void bind() = 0;
        virtual void unbind() = 0;
        virtual void setMVP(const glm::mat4 &mvp) = 0;
        virtual void setColor(const Color &color) = 0;

        static GridShader *create();
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
}

#endif