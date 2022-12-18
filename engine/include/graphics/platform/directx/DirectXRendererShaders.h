#ifndef DIRECTX_RENDERER_SHADERS_H__
#define DIRECTX_RENDERER_SHADERS_H__

#include "../../RendererShaders.h"

namespace PhysicsEngine
{
    class DirectXRendererShaders : public RendererShaders
    {
    private:
        ShaderProgram *mSSAOShader;
        ShaderProgram *mGeometryShader;
        ShaderProgram *mDepthShader;
        ShaderProgram *mDepthCubemapShader;
        ShaderProgram *mScreenQuadShader;
        ShaderProgram *mSpriteShader;
        ShaderProgram *mGBufferShader;
        ShaderProgram *mColorShader;
        ShaderProgram *mColorInstancedShader;
        ShaderProgram *mNormalShader;
        ShaderProgram *mNormalInstancedShader;
        ShaderProgram *mPositionShader;
        ShaderProgram *mPositionInstancedShader;
        ShaderProgram *mLinearDepthShader;
        ShaderProgram *mLinearDepthInstancedShader;
        ShaderProgram *mLineShader;
        ShaderProgram *mGizmoShader;
        ShaderProgram *mGridShader;

    public:
        DirectXRendererShaders();
        ~DirectXRendererShaders();

    protected:
        ShaderProgram *getSSAOShader_impl() override;
        ShaderProgram *getGeometryShader_impl() override;
        ShaderProgram *getDepthShader_impl() override;
        ShaderProgram *getDepthCubemapShader_impl() override;
        ShaderProgram *getScreenQuadShader_impl() override;
        ShaderProgram *getSpriteShader_impl() override;
        ShaderProgram *getGBufferShader_impl() override;
        ShaderProgram *getColorShader_impl() override;
        ShaderProgram *getColorInstancedShader_impl() override;
        ShaderProgram *getNormalShader_impl() override;
        ShaderProgram *getNormalInstancedShader_impl() override;
        ShaderProgram *getPositionShader_impl() override;
        ShaderProgram *getPositionInstancedShader_impl() override;
        ShaderProgram *getLinearDepthShader_impl() override;
        ShaderProgram *getLinearDepthInstancedShader_impl() override;
        ShaderProgram *getLineShader_impl() override;
        ShaderProgram *getGizmoShader_impl() override;
        ShaderProgram *getGridShader_impl() override;

        std::string getStandardVertexShader_impl() override;
        std::string getStandardFragmentShader_impl() override;
    };
}

#endif
