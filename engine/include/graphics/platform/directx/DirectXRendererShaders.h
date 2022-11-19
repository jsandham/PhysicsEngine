#ifndef DIRECTX_RENDERER_SHADERS_H__
#define DIRECTX_RENDERER_SHADERS_H__

#include "../../RendererShaders.h"

namespace PhysicsEngine
{
    class DirectXRendererShaders : public RendererShaders
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

    protected:
        void init_impl() override;

        /*void compileSSAOShader_impl() override;
        void compileGeometryShader_impl() override;
        void compileDepthShader_impl() override;
        void compileDepthCubemapShader_impl() override;
        void compileScreenQuadShader_impl() override;
        void compileSpriteShader_impl() override;
        void compileGBufferShader_impl() override;
        void compileColorShader_impl() override;
        void compileColorInstancedShader_impl() override;
        void compileNormalShader_impl() override;
        void compileNormalInstancedShader_impl() override;
        void compilePositionShader_impl() override;
        void compilePositionInstancedShader_impl() override;
        void compileLinearDepthShader_impl() override;
        void compileLinearDepthInstancedShader_impl() override;
        void compileLineShader_impl() override;
        void compileGizmoShader_impl() override;
        void compileGridShader_impl() override;*/

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
