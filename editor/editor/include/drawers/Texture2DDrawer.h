#ifndef __TEXTURE2D_DRAWER_H__
#define __TEXTURE2D_DRAWER_H__

#include "../EditorClipboard.h"
#include "InspectorDrawer.h"

namespace PhysicsEditor
{
class Texture2DDrawer : public InspectorDrawer
{
  private:
    GLuint mFBO;
    GLuint mColor;
    GLuint mDepth;

    GLuint mVAO;
    GLuint mVBO;

    GLuint mProgramR;
    GLuint mProgramG;
    GLuint mProgramB;
    GLuint mProgramA;
    int mTexLocR;
    int mTexLocG;
    int mTexLocB;
    int mTexLocA;

    Guid mCurrentTexId;
    GLint mDrawTex;

  public:
    Texture2DDrawer();
    ~Texture2DDrawer();

    virtual void render(Clipboard &clipboard, Guid id) override;
};
} // namespace PhysicsEditor

#endif