#ifndef OPENGL_RENDERER_SHADERS_H__
#define OPENGL_RENDERER_SHADERS_H__

#include "../../RendererShaders.h"

namespace PhysicsEngine
{
    class OpenGLRendererShaders : public RendererShaders
    {
    private:
        SSAOShader mSSAOShader;
        GeometryShader mGeometryShader;
        DepthShader mDepthShader;
        DepthCubemapShader mDepthCubemapShader;
        ScreenQuadShader mScreenQuadShader;
        SpriteShader mSpriteShader;
        GBufferShader mGBufferShader;
        ColorShader mColorShader;
        ColorInstancedShader mColorInstancedShader;
        NormalShader mNormalShader;
        NormalInstancedShader mNormalInstancedShader;
        PositionShader mPositionShader;
        PositionInstancedShader mPositionInstancedShader;
        LinearDepthShader mLinearDepthShader;
        LinearDepthInstancedShader mLinearDepthInstancedShader;
        LineShader mLineShader;
        GizmoShader mGizmoShader;
        GridShader mGridShader;

        void compileSSAOShader();
        void compileGeometryShader();
        void compileDepthShader();
        void compileDepthCubemapShader();
        void compileScreenQuadShader();
        void compileSpriteShader();
        void compileGBufferShader();
        void compileColorShader();
        void compileColorInstancedShader();
        void compileNormalShader();
        void compileNormalInstancedShader();
        void compilePositionShader();
        void compilePositionInstancedShader();
        void compileLinearDepthShader();
        void compileLinearDepthInstancedShader();
        void compileLineShader();
        void compileGizmoShader();
        void compileGridShader();

    protected:
        void init_impl() override;

        SSAOShader getSSAOShader_impl() override;
        GeometryShader getGeometryShader_impl() override;
        DepthShader getDepthShader_impl() override;
        DepthCubemapShader getDepthCubemapShader_impl() override;
        ScreenQuadShader getScreenQuadShader_impl() override;
        SpriteShader getSpriteShader_impl() override;
        GBufferShader getGBufferShader_impl() override;
        ColorShader getColorShader_impl() override;
        ColorInstancedShader getColorInstancedShader_impl() override;
        NormalShader getNormalShader_impl() override;
        NormalInstancedShader getNormalInstancedShader_impl() override;
        PositionShader getPositionShader_impl() override;
        PositionInstancedShader getPositionInstancedShader_impl() override;
        LinearDepthShader getLinearDepthShader_impl() override;
        LinearDepthInstancedShader getLinearDepthInstancedShader_impl() override;
        LineShader getLineShader_impl() override;
        GizmoShader getGizmoShader_impl() override;
        GridShader getGridShader_impl() override;

        std::string getStandardVertexShader_impl() override;
        std::string getStandardFragmentShader_impl() override;
    };
}

#endif