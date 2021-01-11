#ifndef __ABOUT_POPUP_H__
#define __ABOUT_POPUP_H__

#include "PopupWindow.h"

namespace PhysicsEditor
{
class AboutPopup : public PopupWindow
{
  public:
    AboutPopup();
    ~AboutPopup();
    AboutPopup(const AboutPopup &other) = delete;
    AboutPopup &operator=(const AboutPopup &other) = delete;

    void init(EditorClipboard &clipboard) override;
    void update(EditorClipboard &clipboard) override;
};
} // namespace PhysicsEditor

#endif