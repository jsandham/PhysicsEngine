#include "../../include/drawers/RigidbodyDrawer.h"

#include "components/Rigidbody.h"

#include "imgui.h"

using namespace PhysicsEditor;

RigidbodyDrawer::RigidbodyDrawer()
{
}

RigidbodyDrawer::~RigidbodyDrawer()
{
}

void RigidbodyDrawer::render(Clipboard &clipboard, const Guid& id)
{
    InspectorDrawer::render(clipboard, id);

    ImGui::Separator();
    mContentMin = ImGui::GetItemRectMin();

    if (ImGui::TreeNodeEx("Rigidbody", ImGuiTreeNodeFlags_DefaultOpen))
    {
        Rigidbody *rigidbody = clipboard.getWorld()->getActiveScene()->getComponentByGuid<Rigidbody>(id);

        if (rigidbody != nullptr)
        {
            ImGui::Text(("ComponentId: " + id.toString()).c_str());

            bool useGravity = rigidbody->mUseGravity;
            float mass = rigidbody->mMass;
            float drag = rigidbody->mDrag;
            float angularDrag = rigidbody->mAngularDrag;

            if (ImGui::Checkbox("Use Gravity", &useGravity))
            {
                rigidbody->mUseGravity = useGravity;
            }

            if (ImGui::InputFloat("Mass", &mass))
            {
                rigidbody->mMass = mass;
            }

            if (ImGui::InputFloat("Drag", &drag))
            {
                rigidbody->mDrag = drag;
            }

            if (ImGui::InputFloat("Angular Drag", &angularDrag))
            {
                rigidbody->mAngularDrag = angularDrag;
            }

            bool enabled = rigidbody->mEnabled;
            if (ImGui::Checkbox("Enabled?", &enabled))
            {
                rigidbody->mEnabled = enabled;
            }
        }
        
        ImGui::TreePop();
    }

    ImGui::Separator();
    mContentMax = ImGui::GetItemRectMax();
}