#ifndef __CONSOLE_H__
#define __CONSOLE_H__

#include "Window.h"

namespace PhysicsEditor
{
class Console : public Window
{
  public:
    Console();
    ~Console();
    Console(const Console &other) = delete;
    Console &operator=(const Console &other) = delete;

    void init(EditorClipboard &clipboard);
    void update(EditorClipboard &clipboard, bool isOpenedThisFrame);
};
} // namespace PhysicsEditor

#endif
