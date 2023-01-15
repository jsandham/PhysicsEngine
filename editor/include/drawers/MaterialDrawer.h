#ifndef MATERIAL_DRAWER_H__
#define MATERIAL_DRAWER_H__

#include "InspectorDrawer.h"

#include "core/Material.h"

#include <graphics/Framebuffer.h>
#include <graphics/RendererUniforms.h>

namespace PhysicsEditor
{
class MaterialDrawer : public InspectorDrawer
{
  private:
    Framebuffer* mFBO;

    CameraUniform* mCameraUniform;
    LightUniform* mLightUniform;

    glm::vec3 mCameraPos;
    glm::mat4 mModel;
    glm::mat4 mView;
    glm::mat4 mProjection;
    glm::mat4 mViewProjection;

  public:
    MaterialDrawer();
    ~MaterialDrawer();

    virtual void render(Clipboard &clipboard, const Guid& id) override;

  private:
      void drawIntUniform(Clipboard& clipboard, Material* material, ShaderUniform* uniform);
      void drawFloatUniform(Clipboard& clipboard, Material* material, ShaderUniform* uniform);
      void drawColorUniform(Clipboard& clipboard, Material* material, ShaderUniform* uniform);
      void drawVec2Uniform(Clipboard& clipboard, Material* material, ShaderUniform* uniform);
      void drawVec3Uniform(Clipboard& clipboard, Material* material, ShaderUniform* uniform);
      void drawVec4Uniform(Clipboard& clipboard, Material* material, ShaderUniform* uniform);
      void drawTexture2DUniform(Clipboard& clipboard, Material* material, ShaderUniform* uniform);
      void drawCubemapUniform(Clipboard& clipboard, Material* material, ShaderUniform* uniform);
};

} // namespace PhysicsEditor

#endif