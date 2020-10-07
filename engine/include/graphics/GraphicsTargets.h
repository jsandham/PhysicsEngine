#ifndef __GRAPHICS_TARGETS_H__
#define __GRAPHICS_TARGETS_H__

#include <GL/glew.h>
#include <gl/gl.h>

namespace PhysicsEngine
{
typedef struct GraphicsTargets
{
    GLuint mMainFBO;
    GLuint mColorTex;
    GLuint mDepthTex;

    GLuint mColorPickingFBO;
    GLuint mColorPickingTex;
    GLuint mColorPickingDepthTex;

    GLuint mGeometryFBO;
    GLuint mPositionTex;
    GLuint mNormalTex;
    GLuint mAlbedoSpecTex;

    GLuint mSsaoFBO;
    GLuint mSsaoColorTex;
    GLuint mSsaoNoiseTex;

    GraphicsTargets()
    {
        mMainFBO = 0;
        mColorTex = 0;
        mDepthTex = 0;

        mColorPickingFBO = 0;
        mColorPickingTex = 0;
        mColorPickingDepthTex = 0;

        mGeometryFBO = 0;
        mPositionTex = 0;
        mNormalTex = 0;
        mAlbedoSpecTex = 0;

        mSsaoFBO = 0;
        mSsaoColorTex = 0;
        mSsaoNoiseTex = 0;
    }
} GraphicsTargets;
} // namespace PhysicsEngine

#endif
