#include "../../include/drawers/CapsuleColliderDrawer.h"

#include "components/CapsuleCollider.h"

#include "imgui.h"

using namespace PhysicsEditor;

CapsuleColliderDrawer::CapsuleColliderDrawer()
{
}

CapsuleColliderDrawer::~CapsuleColliderDrawer()
{
}

void CapsuleColliderDrawer::render(Clipboard &clipboard, const Guid& id)
{
    InspectorDrawer::render(clipboard, id);

    ImGui::Separator();
    mContentMin = ImGui::GetItemRectMin();

    if (ImGui::TreeNodeEx("CapsuleCollider", ImGuiTreeNodeFlags_DefaultOpen))
    {
        CapsuleCollider *capsuleCollider = clipboard.getWorld()->getActiveScene()->getComponentByGuid<CapsuleCollider>(id);

        if (capsuleCollider != nullptr)
        {
            ImGui::Text(("ComponentId: " + id.toString()).c_str());

            if (ImGui::TreeNode("Capsule"))
            {
                float centre[3];
                centre[0] = capsuleCollider->mCapsule.mCentre.x;
                centre[1] = capsuleCollider->mCapsule.mCentre.y;
                centre[2] = capsuleCollider->mCapsule.mCentre.z;

                ImGui::InputFloat3("Centre", &centre[0]);
                ImGui::InputFloat("Radius", &capsuleCollider->mCapsule.mRadius);
                ImGui::InputFloat("Height", &capsuleCollider->mCapsule.mHeight);

                capsuleCollider->mCapsule.mCentre.x = centre[0];
                capsuleCollider->mCapsule.mCentre.y = centre[1];
                capsuleCollider->mCapsule.mCentre.z = centre[2];

                ImGui::TreePop();
            }

            bool enabled = capsuleCollider->mEnabled;
            if (ImGui::Checkbox("Enabled?", &enabled))
            {
                capsuleCollider->mEnabled = enabled;
            }
        }

        ImGui::TreePop();
    }

    ImGui::Separator();
    mContentMax = ImGui::GetItemRectMax();
}