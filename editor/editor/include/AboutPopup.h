#ifndef __ABOUT_POPUP_H__
#define __ABOUT_POPUP_H__

namespace PhysicsEditor
{
class AboutPopup
{
  private:
    bool isVisible;

  public:
    AboutPopup();
    ~AboutPopup();

    void render(bool becomeVisibleThisFrame);
};
} // namespace PhysicsEditor

#endif