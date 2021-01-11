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

    void init(EditorClipboard &clipboard) override;
    void update(EditorClipboard &clipboard) override;
};
} // namespace PhysicsEditor

#endif
