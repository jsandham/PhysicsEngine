#include "../../include/drawers/BoxColliderDrawer.h"
#include "../../include/Undo.h"
#include "../../include/EditorCommands.h"

#include "components/BoxCollider.h"

#include "imgui.h"
#include "imgui_impl_opengl3.h"
#include "imgui_impl_win32.h"
#include "imgui_internal.h"

using namespace PhysicsEditor;

BoxColliderDrawer::BoxColliderDrawer()
{
}

BoxColliderDrawer::~BoxColliderDrawer()
{
}

void BoxColliderDrawer::render(EditorClipboard &clipboard, Guid id)
{
    if (ImGui::TreeNodeEx("BoxCollider", ImGuiTreeNodeFlags_DefaultOpen))
    {
        BoxCollider *boxCollider = clipboard.getWorld()->getComponentById<BoxCollider>(id);

        ImGui::Text(("EntityId: " + boxCollider->getEntityId().toString()).c_str());
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

        ImGui::TreePop();
    }
}