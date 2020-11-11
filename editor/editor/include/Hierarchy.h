#ifndef __HIERARCHY_H__
#define __HIERARCHY_H__

#include <set>
#include <string>
#include <vector>

#include "core/Entity.h"
#include "core/World.h"

#include "../include/EditorClipboard.h"
#include "../include/EditorScene.h"

using namespace PhysicsEngine;

namespace PhysicsEditor
{
class Hierarchy
{
  private:
    struct HierarchyEntry
    {
        Entity* entity;
        std::string label;
        int indentLevel;
    };

    std::vector<HierarchyEntry> entries;
    std::unordered_map<Guid, int> idToEntryIndex;

    bool rebuildRequired;

  public:
    Hierarchy();
    ~Hierarchy();

    void render(World *world, EditorScene &scene, EditorClipboard &clipboard, const std::set<Guid> &editorOnlyEntityIds,
                bool isOpenedThisFrame);

  private:
    void rebuildEntityLists(World* world, const std::set<Guid>& editorOnlyEntityIds);
};
} // namespace PhysicsEditor

#endif