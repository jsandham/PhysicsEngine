#ifndef __HIERARCHY_H__
#define __HIERARCHY_H__

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
    struct HierarchyEntry
    {
        PhysicsEngine::Entity *entity;
        std::string label;
        int indentLevel;
    };

    std::vector<HierarchyEntry> mEntries;
    std::unordered_map<PhysicsEngine::Guid, int> mIdToEntryIndex;

  public:
    Hierarchy();
    ~Hierarchy();
    Hierarchy(const Hierarchy &other) = delete;
    Hierarchy &operator=(const Hierarchy &other) = delete;

    void init(Clipboard &clipboard) override;
    void update(Clipboard &clipboard) override;

  private:
    void rebuildEntityLists(PhysicsEngine::World *world);
};
} // namespace PhysicsEditor

#endif