#ifndef WORLD_DEFAULT_ASSETS_H__
#define WORLD_DEFAULT_ASSETS_H__

#include "Guid.h"

namespace PhysicsEngine
{
class World;

// Default assets that all worlds have access to
struct WorldDefaultAssets
{
    // default loaded meshes
    Guid mSphereMeshId;
    Guid mCubeMeshId;
    Guid mPlaneMeshId;

    // default loaded shaders
    Guid mStandardShaderId;
    Guid mColorLitShaderId;
    Guid mNormalShaderId;
    Guid mTangentShaderId;
    Guid mBinormalShaderId;

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
    Guid mStandardDeferedShaderId;

    // default loaded materials
    Guid mSimpleLitMaterialId;
    Guid mColorMaterialId;

    void loadInternalAssets(World* world);
};
} // namespace PhysicsEngine

#endif
