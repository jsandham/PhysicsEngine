#ifndef __TEXTURE2D_DRAWER_H__
#define __TEXTURE2D_DRAWER_H__

#include "../EditorClipboard.h"
#include "InspectorDrawer.h"

namespace PhysicsEditor
{
class Texture2DDrawer : public InspectorDrawer
{
  private:
    unsigned int mFBO;
    unsigned int mColor;
    unsigned int mDepth;

    unsigned int mVAO;
    unsigned int mVBO;

    unsigned int mProgramR;
    unsigned int mProgramG;
    unsigned int mProgramB;
    unsigned int mProgramA;

    int mTexLocR;
    int mTexLocG;
    int mTexLocB;
    int mTexLocA;

    Guid mCurrentTexId;
    int mDrawTex;

  public:
    Texture2DDrawer();
    ~Texture2DDrawer();

    virtual void render(Clipboard &clipboard, Guid id) override;
};
} // namespace PhysicsEditor

#endif