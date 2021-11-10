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
      float mWindowWidth;
      float mWindowHeight;
      ImVec2 mContentMin;
      ImVec2 mContentMax;
      bool mOpen;
      bool mFocused;
      bool mHovered;
      bool mOpenedLastFrame;
      bool mFocusedLastFrame;
      bool mHoveredLastFrame;

  public:
    Window();
    Window(const std::string name);
    virtual ~Window() = 0;

    void draw(Clipboard& clipboard, bool isOpenedThisFrame, float alpha = 1.0f, ImGuiWindowFlags flags = ImGuiWindowFlags_None);

    virtual void init(Clipboard &clipboard) = 0;
    virtual void update(Clipboard &clipboard) = 0;

    void close();

    ImVec2 getWindowPos() const;
    float getWindowWidth() const;
    float getWindowHeight() const;
    ImVec2 getContentMin() const;
    ImVec2 getContentMax() const;
    bool isOpen() const;
    bool isFocused() const;
    bool isHovered() const;
    bool openedThisFrame() const;
    bool closedThisFrame() const;
    bool focusedThisFrame() const;
    bool hoveredThisFrame() const;
    bool unfocusedThisFrame() const;
    bool unhoveredThisFrame() const;

};
} // namespace PhysicsEditor

#endif