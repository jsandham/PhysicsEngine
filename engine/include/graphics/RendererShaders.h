#ifndef RENDERER_SHADERS_H__
#define RENDERER_SHADERS_H__

#include <string>

namespace PhysicsEngine
{
    struct SSAOShader
    {
        int mProgram;
        int mProjectionLoc;
        int mPositionTexLoc;
        int mNormalTexLoc;
        int mNoiseTexLoc;
        int mSamplesLoc[64];
    };

    struct GeometryShader
    {
        int mProgram;
        int mModelLoc;
    };

    struct DepthShader
    {
        int mProgram;
        int mModelLoc;
        int mViewLoc;
        int mProjectionLoc;
    };

    struct DepthCubemapShader
    {
        int mProgram;
        int mLightPosLoc;
        int mFarPlaneLoc;
        int mModelLoc;
        int mCubeViewProjMatricesLoc0;
        int mCubeViewProjMatricesLoc1;
        int mCubeViewProjMatricesLoc2;
        int mCubeViewProjMatricesLoc3;
        int mCubeViewProjMatricesLoc4;
        int mCubeViewProjMatricesLoc5;
    };

    struct ScreenQuadShader
    {
        int mProgram;
        int mTexLoc;
    };

    struct SpriteShader
    {
        int mProgram;
        int mModelLoc;
        int mViewLoc;
        int mProjectionLoc;
        int mColorLoc;
        int mImageLoc;
    };

    struct GBufferShader
    {
        int mProgram;
        int mModelLoc;
        int mDiffuseTexLoc;
        int mSpecTexLoc;
    };

    struct ColorShader
    {
        int mProgram;
        int mModelLoc;
        int mColorLoc;
    };

    struct ColorInstancedShader
    {
        int mProgram;
    };

    struct NormalShader
    {
        int mProgram;
        int mModelLoc;
    };

    struct NormalInstancedShader
    {
        int mProgram;
    };

    struct PositionShader
    {
        int mProgram;
        int mModelLoc;
    };

    struct PositionInstancedShader
    {
        int mProgram;
    };

    struct LinearDepthShader
    {
        int mProgram;
        int mModelLoc;
    };

    struct LinearDepthInstancedShader
    {
        int mProgram;
    };

    struct LineShader
    {
        int mProgram;
        int mMVPLoc;
    };

    struct GizmoShader
    {
        int mProgram;
        int mModelLoc;
        int mViewLoc;
        int mProjLoc;
        int mColorLoc;
        int mLightPosLoc;
    };

    struct GridShader
    {
        int mProgram;
        int mMVPLoc;
        int mColorLoc;
    };

    class RendererShaders
    {
    private:
        static RendererShaders* sInstance;

    public:
        static void init();
        static RendererShaders* getRendererShaders();

        static SSAOShader getSSAOShader();
        static GeometryShader getGeometryShader();
        static DepthShader getDepthShader();
        static DepthCubemapShader getDepthCubemapShader();
        static ScreenQuadShader getScreenQuadShader();
        static SpriteShader getSpriteShader();
        static GBufferShader getGBufferShader();
        static ColorShader getColorShader();
        static ColorInstancedShader getColorInstancedShader();
        static NormalShader getNormalShader();
        static NormalInstancedShader getNormalInstancedShader();
        static PositionShader getPositionShader();
        static PositionInstancedShader getPositionInstancedShader();
        static LinearDepthShader getLinearDepthShader();
        static LinearDepthInstancedShader getLinearDepthInstancedShader();
        static LineShader getLineShader();
        static GizmoShader getGizmoShader();
        static GridShader getGridShader();

        static std::string getStandardVertexShader();
        static std::string getStandardFragmentShader();

    protected:
        virtual void init_impl() = 0;

        virtual SSAOShader getSSAOShader_impl() = 0;
        virtual GeometryShader getGeometryShader_impl() = 0;
        virtual DepthShader getDepthShader_impl() = 0;
        virtual DepthCubemapShader getDepthCubemapShader_impl() = 0;
        virtual ScreenQuadShader getScreenQuadShader_impl() = 0;
        virtual SpriteShader getSpriteShader_impl() = 0;
        virtual GBufferShader getGBufferShader_impl() = 0;
        virtual ColorShader getColorShader_impl() = 0;
        virtual ColorInstancedShader getColorInstancedShader_impl() = 0;
        virtual NormalShader getNormalShader_impl() = 0;
        virtual NormalInstancedShader getNormalInstancedShader_impl() = 0;
        virtual PositionShader getPositionShader_impl() = 0;
        virtual PositionInstancedShader getPositionInstancedShader_impl() = 0;
        virtual LinearDepthShader getLinearDepthShader_impl() = 0;
        virtual LinearDepthInstancedShader getLinearDepthInstancedShader_impl() = 0;
        virtual LineShader getLineShader_impl() = 0;
        virtual GizmoShader getGizmoShader_impl() = 0;
        virtual GridShader getGridShader_impl() = 0;

        virtual std::string getStandardVertexShader_impl() = 0;
        virtual std::string getStandardFragmentShader_impl() = 0;
    };
}

#endif