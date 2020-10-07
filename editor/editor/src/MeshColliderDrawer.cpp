#include "../include/MeshColliderDrawer.h"

#include "components/MeshCollider.h"

#include "imgui.h"
#include "imgui_impl_opengl3.h"
#include "imgui_impl_win32.h"
#include "imgui_internal.h"

using namespace PhysicsEditor;

MeshColliderDrawer::MeshColliderDrawer()
{
}

MeshColliderDrawer::~MeshColliderDrawer()
{
}

void MeshColliderDrawer::render(World *world, EditorProject &project, EditorScene &scene, EditorClipboard &clipboard,
                                Guid id)
{
    if (ImGui::TreeNodeEx("MeshCollider", ImGuiTreeNodeFlags_DefaultOpen))
    {
        MeshCollider *meshCollider = world->getComponentById<MeshCollider>(id);

        ImGui::Text(("EntityId: " + meshCollider->getEntityId().toString()).c_str());
        ImGui::Text(("ComponentId: " + id.toString()).c_str());

        ImGui::TreePop();
    }
}