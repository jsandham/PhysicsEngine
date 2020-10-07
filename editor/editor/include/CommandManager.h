#ifndef __COMMAND_MANAGER_H__
#define __COMMAND_MANAGER_H__

#include <queue>
#include <vector>

#include "Command.h"

namespace PhysicsEditor
{
class CommandManager
{
  private:
    static int counter;
    static std::vector<Command *> commandHistory;
    static std::queue<Command *> commandQueue;

  public:
    CommandManager();
    ~CommandManager();

    void update();

    static void addCommand(Command *command);
    static void executeCommand();
    static void undoCommand();
    static bool canUndo();
    static bool canRedo();
};
} // namespace PhysicsEditor

#endif
