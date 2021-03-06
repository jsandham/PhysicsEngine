#ifndef __WINDOW_H__
#define __WINDOW_H__

#include <string>

#include "../EditorClipboard.h"

namespace PhysicsEditor
{
class Window
{
  private:
      std::string mName;
      bool mOpen;

  public:
    Window();
    Window(const std::string name);
    virtual ~Window() = 0;

    void draw(EditorClipboard& clipboard, bool isOpenedThisFrame);

    virtual void init(EditorClipboard &clipboard) = 0;
    virtual void update(EditorClipboard &clipboard) = 0;
};
} // namespace PhysicsEditor

#endif