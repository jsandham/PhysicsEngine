#ifndef HIERARCHY_H__
#define HIERARCHY_H__

#include <set>
#include <string>
#include <vector>

#include "imgui.h"

#include "core/Entity.h"
#include "core/World.h"

#include "../EditorClipboard.h"

namespace PhysicsEditor
{
class Hierarchy
{
  private:
    std::vector<int> mEntries;

    ImVec2 mWindowPos;
    ImVec2 mContentMin;
    ImVec2 mContentMax;

    bool mOpen;

  public:
    Hierarchy();
    ~Hierarchy();
    Hierarchy(const Hierarchy &other) = delete;
    Hierarchy &operator=(const Hierarchy &other) = delete;

    void init(Clipboard& clipboard);
    void update(Clipboard &clipboard, bool isOpenedThisFrame);

    ImVec2 getWindowPos() const;
    ImVec2 getContentMin() const;
    ImVec2 getContentMax() const;
};
} // namespace PhysicsEditor

#endif