#include "../../include/drawers/SphereColliderDrawer.h"
#include "../../include/Undo.h"
#include "../../include/EditorCommands.h"

#include "components/SphereCollider.h"

#include "imgui.h"
#include "imgui_impl_opengl3.h"
#include "imgui_impl_win32.h"
#include "imgui_internal.h"

using namespace PhysicsEditor;

SphereColliderDrawer::SphereColliderDrawer()
{
}

SphereColliderDrawer::~SphereColliderDrawer()
{
}

void SphereColliderDrawer::render(EditorClipboard &clipboard, Guid id)
{
    if (ImGui::TreeNodeEx("SphereCollider", ImGuiTreeNodeFlags_DefaultOpen))
    {
        SphereCollider *sphereCollider = clipboard.getWorld()->getComponentById<SphereCollider>(id);

        ImGui::Text(("EntityId: " + sphereCollider->getEntityId().toString()).c_str());
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

        ImGui::TreePop();
    }
}