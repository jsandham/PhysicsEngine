#include "../../include/drawers/BoxColliderDrawer.h"
#include "../../include/Undo.h"
#include "../../include/EditorCommands.h"

#include "components/BoxCollider.h"

#include "imgui.h"

using namespace PhysicsEditor;

BoxColliderDrawer::BoxColliderDrawer()
{
}

BoxColliderDrawer::~BoxColliderDrawer()
{
}

void BoxColliderDrawer::render(Clipboard &clipboard, Guid id)
{
    InspectorDrawer::render(clipboard, id);

    ImGui::Separator();
    mContentMin = ImGui::GetItemRectMin();

    if (ImGui::TreeNodeEx("BoxCollider", ImGuiTreeNodeFlags_DefaultOpen))
    {
        BoxCollider *boxCollider = clipboard.getWorld()->getComponentById<BoxCollider>(id);

        if (boxCollider != nullptr)
        {
            ImGui::Text(("ComponentId: " + id.toString()).c_str());

            if (ImGui::TreeNode("Bounds"))
            {
                glm::vec3 centre = boxCollider->mAABB.mCentre;
                glm::vec3 size = boxCollider->mAABB.mSize;

                if (ImGui::InputFloat3("Centre", glm::value_ptr(centre)))
                {
                    Undo::recordComponent(boxCollider);

                    boxCollider->mAABB.mCentre = centre;
                }
                if (ImGui::InputFloat3("Size", glm::value_ptr(size)))
                {
                    Undo::recordComponent(boxCollider);

                    boxCollider->mAABB.mSize = size;
                }

                ImGui::TreePop();
            }
        }

        ImGui::TreePop();
    }

    ImGui::Separator();
    mContentMax = ImGui::GetItemRectMax();
}