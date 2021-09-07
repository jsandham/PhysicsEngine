#include "../../include/drawers/SphereColliderDrawer.h"
#include "../../include/Undo.h"
#include "../../include/EditorCommands.h"

#include "components/SphereCollider.h"

#include "imgui.h"

using namespace PhysicsEditor;

SphereColliderDrawer::SphereColliderDrawer()
{
}

SphereColliderDrawer::~SphereColliderDrawer()
{
}

void SphereColliderDrawer::render(Clipboard &clipboard, Guid id)
{
    InspectorDrawer::render(clipboard, id);

    ImGui::Separator();
    mContentMin = ImGui::GetItemRectMin();

    if (ImGui::TreeNodeEx("SphereCollider", ImGuiTreeNodeFlags_DefaultOpen))
    {
        SphereCollider *sphereCollider = clipboard.getWorld()->getComponentById<SphereCollider>(id);

        if (sphereCollider != nullptr)
        {
            ImGui::Text(("ComponentId: " + id.toString()).c_str());

            if (ImGui::TreeNode("Sphere"))
            {
                glm::vec3 centre = sphereCollider->mSphere.mCentre;
                float radius = sphereCollider->mSphere.mRadius;

                if (ImGui::InputFloat3("Centre", glm::value_ptr(centre)))
                {
                    Undo::recordComponent(sphereCollider);

                    sphereCollider->mSphere.mCentre = centre;
                }
                if (ImGui::InputFloat("Radius", &radius))
                {
                    Undo::recordComponent(sphereCollider);

                    sphereCollider->mSphere.mRadius = radius;
                }

                ImGui::TreePop();
            }

            bool enabled = sphereCollider->mEnabled;
            if (ImGui::Checkbox("Enabled?", &enabled))
            {
                sphereCollider->mEnabled = enabled;
            }
        }

        ImGui::TreePop();
    }

    ImGui::Separator();
    mContentMax = ImGui::GetItemRectMax();
}