#ifndef CONSOLE_H__
#define CONSOLE_H__

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

    void init(Clipboard &clipboard) override;
    void update(Clipboard &clipboard) override;
};
} // namespace PhysicsEditor

#endif
