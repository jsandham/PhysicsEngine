#include "../../include/drawers/MeshColliderDrawer.h"

#include "components/MeshCollider.h"

#include "imgui.h"

using namespace PhysicsEditor;

MeshColliderDrawer::MeshColliderDrawer()
{
}

MeshColliderDrawer::~MeshColliderDrawer()
{
}

void MeshColliderDrawer::render(Clipboard &clipboard, Guid id)
{
    if (ImGui::TreeNodeEx("MeshCollider", ImGuiTreeNodeFlags_DefaultOpen))
    {
        MeshCollider *meshCollider = clipboard.getWorld()->getComponentById<MeshCollider>(id);

        ImGui::Text(("EntityId: " + meshCollider->getEntityId().toString()).c_str());
        ImGui::Text(("ComponentId: " + id.toString()).c_str());

        ImGui::TreePop();
    }
}