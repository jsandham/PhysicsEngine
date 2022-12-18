#ifndef RENDERER_SHADERS_H__
#define RENDERER_SHADERS_H__

#include <string>
#include "ShaderProgram.h"

namespace PhysicsEngine
{
    class RendererShaders
    {
    private:
        static RendererShaders* sInstance;

    public:
        static void init();
        static RendererShaders* getRendererShaders();

        static ShaderProgram *getSSAOShader();
        static ShaderProgram *getGeometryShader();
        static ShaderProgram *getDepthShader();
        static ShaderProgram *getDepthCubemapShader();
        static ShaderProgram *getScreenQuadShader();
        static ShaderProgram *getSpriteShader();
        static ShaderProgram *getGBufferShader();
        static ShaderProgram *getColorShader();
        static ShaderProgram *getColorInstancedShader();
        static ShaderProgram *getNormalShader();
        static ShaderProgram *getNormalInstancedShader();
        static ShaderProgram *getPositionShader();
        static ShaderProgram *getPositionInstancedShader();
        static ShaderProgram *getLinearDepthShader();
        static ShaderProgram *getLinearDepthInstancedShader();
        static ShaderProgram *getLineShader();
        static ShaderProgram *getGizmoShader();
        static ShaderProgram *getGridShader();

        static std::string getStandardVertexShader();
        static std::string getStandardFragmentShader();

        RendererShaders();
        virtual ~RendererShaders() = 0;

    protected:
        virtual ShaderProgram *getSSAOShader_impl() = 0;
        virtual ShaderProgram *getGeometryShader_impl() = 0;
        virtual ShaderProgram *getDepthShader_impl() = 0;
        virtual ShaderProgram *getDepthCubemapShader_impl() = 0;
        virtual ShaderProgram *getScreenQuadShader_impl() = 0;
        virtual ShaderProgram *getSpriteShader_impl() = 0;
        virtual ShaderProgram *getGBufferShader_impl() = 0;
        virtual ShaderProgram *getColorShader_impl() = 0;
        virtual ShaderProgram *getColorInstancedShader_impl() = 0;
        virtual ShaderProgram *getNormalShader_impl() = 0;
        virtual ShaderProgram *getNormalInstancedShader_impl() = 0;
        virtual ShaderProgram *getPositionShader_impl() = 0;
        virtual ShaderProgram *getPositionInstancedShader_impl() = 0;
        virtual ShaderProgram *getLinearDepthShader_impl() = 0;
        virtual ShaderProgram *getLinearDepthInstancedShader_impl() = 0;
        virtual ShaderProgram *getLineShader_impl() = 0;
        virtual ShaderProgram *getGizmoShader_impl() = 0;
        virtual ShaderProgram *getGridShader_impl() = 0;

        virtual std::string getStandardVertexShader_impl() = 0;
        virtual std::string getStandardFragmentShader_impl() = 0;
    };
}

#endif