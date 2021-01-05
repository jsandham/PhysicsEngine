#ifndef __ABOUT_POPUP_H__
#define __ABOUT_POPUP_H__

#include "Window.h"

namespace PhysicsEditor
{
class AboutPopup : public Window
{
  public:
    AboutPopup();
    ~AboutPopup();
    AboutPopup(const AboutPopup &other) = delete;
    AboutPopup &operator=(const AboutPopup &other) = delete;

    void init(EditorClipboard &clipboard);
    void update(EditorClipboard &clipboard, bool isOpenedThisFrame);
};
} // namespace PhysicsEditor

#endif