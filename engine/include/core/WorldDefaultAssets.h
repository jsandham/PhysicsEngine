#ifndef __WORLD_DEFAULT_ASSETS_H__
#define __WORLD_DEFAULT_ASSETS_H__

#include "Guid.h"

namespace PhysicsEngine
{
// Default assets that all words have access to
struct WorldDefaultAssets
{
    // default loaded meshes
    Guid mSphereMeshId;
    Guid mCubeMeshId;
    Guid mPlaneMeshId;

    // default loaded shaders
    Guid mColorLitShaderId;
    Guid mNormalLitShaderId;
    Guid mTangentLitShaderId;

    Guid mFontShaderId;
    Guid mGizmoShaderId;
    Guid mLineShaderId;
    Guid mColorShaderId;
    Guid mPositionAndNormalsShaderId;
    Guid mSsaoShaderId;
    Guid mScreenQuadShaderId;
    Guid mNormalMapShaderId;
    Guid mDepthMapShaderId;
    Guid mShadowDepthMapShaderId;
    Guid mShadowDepthCubemapShaderId;
    Guid mGbufferShaderId;
    Guid mSimpleLitShaderId;
    Guid mSimpleLitDeferedShaderId;
    Guid mOverdrawShaderId;

    // default loaded materials
    Guid mSimpleLitMaterialId;
    Guid mColorMaterialId;
};
} // namespace PhysicsEngine

#endif
