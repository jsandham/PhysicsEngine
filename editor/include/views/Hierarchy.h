#ifndef HIERARCHY_H__
#define HIERARCHY_H__

#include <set>
#include <string>
#include <vector>

#include "core/Entity.h"
#include "core/World.h"

#include "Window.h"

namespace PhysicsEditor
{
class Hierarchy : public Window
{
  private:
    std::vector<int> mEntries;

  public:
    Hierarchy();
    ~Hierarchy();
    Hierarchy(const Hierarchy &other) = delete;
    Hierarchy &operator=(const Hierarchy &other) = delete;

    void init(Clipboard &clipboard) override;
    void update(Clipboard &clipboard) override;
};
} // namespace PhysicsEditor

#endif