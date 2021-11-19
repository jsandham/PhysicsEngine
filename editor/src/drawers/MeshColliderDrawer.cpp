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

void MeshColliderDrawer::render(Clipboard &clipboard, const Guid& id)
{
    InspectorDrawer::render(clipboard, id);

    ImGui::Separator();
    mContentMin = ImGui::GetItemRectMin();

    if (ImGui::TreeNodeEx("MeshCollider", ImGuiTreeNodeFlags_DefaultOpen))
    {
        MeshCollider *meshCollider = clipboard.getWorld()->getComponentById<MeshCollider>(id);

        if (meshCollider != nullptr)
        {
            ImGui::Text(("ComponentId: " + id.toString()).c_str());

            bool enabled = meshCollider->mEnabled;
            if (ImGui::Checkbox("Enabled?", &enabled))
            {
                meshCollider->mEnabled = enabled;
            }
        }

        ImGui::TreePop();
    }

    ImGui::Separator();
    mContentMax = ImGui::GetItemRectMax();
}