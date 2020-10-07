#ifndef __COMMAND_H__
#define __COMMAND_H__

namespace PhysicsEditor
{
class Command
{
  public:
    Command();
    virtual ~Command() = 0;

    virtual void execute() = 0;
    virtual void undo() = 0;
};
} // namespace PhysicsEditor

#endif
