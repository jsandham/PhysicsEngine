#ifndef TEXTURE2D_DRAWER_H__
#define TEXTURE2D_DRAWER_H__

#include "InspectorDrawer.h"
#include <graphics/Framebuffer.h>
#include <graphics/ShaderProgram.h>
#include <graphics/TextureHandle.h>

namespace PhysicsEditor
{
class Texture2DDrawer : public InspectorDrawer
{
  private:
    Framebuffer* mFBO;

    unsigned int mVAO;
    unsigned int mVBO;

    ShaderProgram* mProgramR;
    ShaderProgram* mProgramG;
    ShaderProgram* mProgramB;
    ShaderProgram* mProgramA;

    Guid mCurrentTexId;
    TextureHandle* mDrawTex;

  public:
    Texture2DDrawer();
    ~Texture2DDrawer();

    virtual void render(Clipboard &clipboard, const Guid& id) override;
};
} // namespace PhysicsEditor

#endif