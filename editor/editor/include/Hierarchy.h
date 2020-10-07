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
    std::vector<Guid> entityIds;
    std::vector<std::string> entityNames;

  public:
    Hierarchy();
    ~Hierarchy();

    void render(World *world, EditorScene &scene, EditorClipboard &clipboard, const std::set<Guid> &editorOnlyEntityIds,
                bool isOpenedThisFrame);
};
} // namespace PhysicsEditor

#endif