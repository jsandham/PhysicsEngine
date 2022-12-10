#ifndef MATERIAL_DRAWER_H__
#define MATERIAL_DRAWER_H__

#include "InspectorDrawer.h"

#include "components/MeshRenderer.h"
#include "core/Material.h"
#include "core/World.h"

#include <graphics/Framebuffer.h>

#include "imgui.h"

#include "../../include/imgui/imgui_extensions.h"

namespace PhysicsEditor
{
class MaterialDrawer : public InspectorDrawer
{
  private:
    //unsigned int mFBO;
    //unsigned int mColor;
    //unsigned int mDepth;
    Framebuffer* mFBO;

    CameraUniform mCameraUniform;
    LightUniform mLightUniform;

    glm::vec3 mCameraPos;
    glm::mat4 mModel;
    glm::mat4 mView;
    glm::mat4 mProjection;

  public:
    MaterialDrawer();
    ~MaterialDrawer();

    virtual void render(Clipboard &clipboard, const Guid& id) override;
};

} // namespace PhysicsEditor

#endif