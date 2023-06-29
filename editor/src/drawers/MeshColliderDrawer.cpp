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

void MeshColliderDrawer::render(Clipboard &clipboard, const PhysicsEngine::Guid& id)
{
    ImGui::Separator();
    mContentMin = ImGui::GetItemRectMin();

    if (ImGui::TreeNodeEx("MeshCollider", ImGuiTreeNodeFlags_DefaultOpen))
    {
        PhysicsEngine::MeshCollider *meshCollider = clipboard.getWorld()->getActiveScene()->getComponentByGuid<PhysicsEngine::MeshCollider>(id);

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

    if (isHovered())
    {
        if (ImGui::BeginPopupContextWindow("RightMouseClickPopup"))
        {
            if (ImGui::MenuItem("RemoveComponent", NULL, false, true))
            {
                PhysicsEngine::MeshCollider* meshCollider = clipboard.getWorld()->getActiveScene()->getComponentByGuid<PhysicsEngine::MeshCollider>(id);
                clipboard.getWorld()->getActiveScene()->immediateDestroyComponent(meshCollider->getEntityGuid(), id, PhysicsEngine::ComponentType<PhysicsEngine::MeshCollider>::type);
            }

            ImGui::EndPopup();
        }
    }
}

bool MeshColliderDrawer::isHovered() const
{
    ImVec2 cursorPos = ImGui::GetMousePos();

    glm::vec2 min = glm::vec2(mContentMin.x, mContentMin.y);
    glm::vec2 max = glm::vec2(mContentMax.x, mContentMax.y);

    PhysicsEngine::Rect rect(min, max);

    return rect.contains(cursorPos.x, cursorPos.y);
}