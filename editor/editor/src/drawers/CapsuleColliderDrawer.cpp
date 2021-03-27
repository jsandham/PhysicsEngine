#include "../../include/drawers/CapsuleColliderDrawer.h"

#include "components/CapsuleCollider.h"

#include "imgui.h"
#include "imgui_impl_opengl3.h"
#include "imgui_impl_win32.h"
#include "imgui_internal.h"

using namespace PhysicsEditor;

CapsuleColliderDrawer::CapsuleColliderDrawer()
{
}

CapsuleColliderDrawer::~CapsuleColliderDrawer()
{
}

void CapsuleColliderDrawer::render(Clipboard &clipboard, Guid id)
{
    if (ImGui::TreeNodeEx("CapsuleCollider", ImGuiTreeNodeFlags_DefaultOpen))
    {
        CapsuleCollider *capsuleCollider = clipboard.getWorld()->getComponentById<CapsuleCollider>(id);

        ImGui::Text(("EntityId: " + capsuleCollider->getEntityId().toString()).c_str());
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

        ImGui::TreePop();
    }
}