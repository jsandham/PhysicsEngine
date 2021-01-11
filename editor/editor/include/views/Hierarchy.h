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

    std::vector<HierarchyEntry> entries;
    std::unordered_map<PhysicsEngine::Guid, int> idToEntryIndex;

    bool rebuildRequired;

  public:
    Hierarchy();
    ~Hierarchy();
    Hierarchy(const Hierarchy &other) = delete;
    Hierarchy &operator=(const Hierarchy &other) = delete;

    void init(EditorClipboard &clipboard) override;
    void update(EditorClipboard &clipboard) override;

  private:
    void rebuildEntityLists(PhysicsEngine::World *world, const std::set<PhysicsEngine::Guid> &editorOnlyEntityIds);
};
} // namespace PhysicsEditor

#endif