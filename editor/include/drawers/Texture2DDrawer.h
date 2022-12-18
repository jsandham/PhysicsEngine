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
    //unsigned int mFBO;
    //unsigned int mColor;
    //unsigned int mDepth;

    unsigned int mVAO;
    unsigned int mVBO;

    ShaderProgram* mProgramR;
    ShaderProgram* mProgramG;
    ShaderProgram* mProgramB;
    ShaderProgram* mProgramA;
    //unsigned int mProgramR;
    //unsigned int mProgramG;
    //unsigned int mProgramB;
    //unsigned int mProgramA;

    //TextureHandle* mTexLocR;
    //TextureHandle* mTexLocG;
    //TextureHandle* mTexLocB;
    //TextureHandle* mTexLocA;
    int mTexLocR;
    int mTexLocG;
    int mTexLocB;
    int mTexLocA;

    Guid mCurrentTexId;
    int mDrawTex;

  public:
    Texture2DDrawer();
    ~Texture2DDrawer();

    virtual void render(Clipboard &clipboard, const Guid& id) override;
};
} // namespace PhysicsEditor

#endif