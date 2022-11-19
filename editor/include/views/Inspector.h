#ifndef INSPECTOR_H__
#define INSPECTOR_H__

#include <vector>

#include "Window.h"

#include "../drawers/InspectorDrawer.h"
#include "../drawers/SceneDrawer.h"
#include "../drawers/MaterialDrawer.h"
#include "../drawers/MeshDrawer.h"
#include "../drawers/ShaderDrawer.h"
#include "../drawers/Texture2DDrawer.h"
#include "../drawers/SpriteDrawer.h"
#include "../drawers/RenderTextureDrawer.h"

#include "../../include/drawers/BoxColliderDrawer.h"
#include "../../include/drawers/CameraDrawer.h"
#include "../../include/drawers/CapsuleColliderDrawer.h"
#include "../../include/drawers/LightDrawer.h"
#include "../../include/drawers/LineRendererDrawer.h"
#include "../../include/drawers/MeshColliderDrawer.h"
#include "../../include/drawers/MeshRendererDrawer.h"
#include "../../include/drawers/SpriteRendererDrawer.h"
#include "../../include/drawers/RigidbodyDrawer.h"
#include "../../include/drawers/SphereColliderDrawer.h"
#include "../../include/drawers/TransformDrawer.h"
#include "../../include/drawers/TerrainDrawer.h"

namespace PhysicsEditor
{
class Inspector : public Window
{
  private:
    SceneDrawer mSceneDrawer;
    MeshDrawer mMeshDrawer;
    MaterialDrawer mMaterialDrawer;
    ShaderDrawer mShaderDrawer;
    Texture2DDrawer mTexture2DDrawer;
    SpriteDrawer mSpriteDrawer;
    RenderTextureDrawer mRenderTextureDrawer;

    TransformDrawer mTransformDrawer;
    RigidbodyDrawer mRigidbodyDrawer;
    CameraDrawer mCameraDrawer;
    MeshRendererDrawer mMeshRendererDrawer;
    SpriteRendererDrawer mSpriteRendererDrawer;
    LineRendererDrawer mLineRendererDrawer;
    LightDrawer mLightDrawer;
    BoxColliderDrawer mBoxColliderDrawer;
    SphereColliderDrawer mSphereColliderDrawer;
    CapsuleColliderDrawer mCapsuleColliderDrawer;
    MeshColliderDrawer mMeshColliderDrawer;
    TerrainDrawer mTerrainDrawer;

  public:
    Inspector();
    ~Inspector();
    Inspector(const Inspector &other) = delete;
    Inspector &operator=(const Inspector &other) = delete;

    void init(Clipboard &clipboard) override;
    void update(Clipboard &clipboard) override;

  private:
    void drawEntity(Clipboard &clipboard);
};
} // namespace PhysicsEditor

#endif