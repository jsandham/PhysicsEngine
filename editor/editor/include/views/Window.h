#ifndef __WINDOW_H__
#define __WINDOW_H__

#include <string>

#include "imgui.h"

#include "../EditorClipboard.h"

namespace PhysicsEditor
{
class Window
{
  private:
      std::string mName;
      ImVec2 mWindowPos;
      ImVec2 mContentMin;
      ImVec2 mContentMax;
      bool mOpen;
      bool mFocused;
      bool mHovered;

  public:
    Window();
    Window(const std::string name);
    virtual ~Window() = 0;

    void draw(Clipboard& clipboard, bool isOpenedThisFrame);

    virtual void init(Clipboard &clipboard) = 0;
    virtual void update(Clipboard &clipboard) = 0;

    ImVec2 getWindowPos() const;
    ImVec2 getContentMin() const;
    ImVec2 getContentMax() const;
    bool isOpen() const;
    bool isFocused() const;
    bool isHovered() const;
};
} // namespace PhysicsEditor

#endif