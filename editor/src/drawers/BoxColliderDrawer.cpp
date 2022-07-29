#include "../../include/drawers/BoxColliderDrawer.h"

#include "components/BoxCollider.h"

#include "imgui.h"

using namespace PhysicsEditor;

BoxColliderDrawer::BoxColliderDrawer()
{
}

BoxColliderDrawer::~BoxColliderDrawer()
{
}

void BoxColliderDrawer::render(Clipboard &clipboard, const Guid& id)
{
    InspectorDrawer::render(clipboard, id);

    ImGui::Separator();
    mContentMin = ImGui::GetItemRectMin();

    if (ImGui::TreeNodeEx("BoxCollider", ImGuiTreeNodeFlags_DefaultOpen))
    {
        BoxCollider *boxCollider = clipboard.getWorld()->getActiveScene()->getComponentByGuid<BoxCollider>(id);

        if (boxCollider != nullptr)
        {
            ImGui::Text(("ComponentId: " + id.toString()).c_str());

            if (ImGui::TreeNode("Bounds"))
            {
                glm::vec3 centre = boxCollider->mAABB.mCentre;
                glm::vec3 size = boxCollider->mAABB.mSize;

                if (ImGui::InputFloat3("Centre", glm::value_ptr(centre)))
                {
                    boxCollider->mAABB.mCentre = centre;
                }
                if (ImGui::InputFloat3("Size", glm::value_ptr(size)))
                {
                    boxCollider->mAABB.mSize = size;
                }

                ImGui::TreePop();
            }

            bool enabled = boxCollider->mEnabled;
            if (ImGui::Checkbox("Enabled?", &enabled))
            {
                boxCollider->mEnabled = enabled;
            }
        }

        ImGui::TreePop();
    }

    ImGui::Separator();
    mContentMax = ImGui::GetItemRectMax();
}