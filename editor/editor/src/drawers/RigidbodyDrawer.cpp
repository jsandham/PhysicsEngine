#include "../../include/drawers/RigidbodyDrawer.h"
#include "../../include/Undo.h"
#include "../../include/EditorCommands.h"

#include "components/Rigidbody.h"

#include "imgui.h"
#include "imgui_impl_opengl3.h"
#include "imgui_impl_win32.h"
#include "imgui_internal.h"

using namespace PhysicsEditor;

RigidbodyDrawer::RigidbodyDrawer()
{
}

RigidbodyDrawer::~RigidbodyDrawer()
{
}

void RigidbodyDrawer::render(EditorClipboard &clipboard, Guid id)
{
    if (ImGui::TreeNodeEx("Rigidbody", ImGuiTreeNodeFlags_DefaultOpen))
    {
        Rigidbody *rigidbody = clipboard.getWorld()->getComponentById<Rigidbody>(id);

        ImGui::Text(("EntityId: " + rigidbody->getEntityId().toString()).c_str());
        ImGui::Text(("ComponentId: " + id.toString()).c_str());

        bool useGravity = rigidbody->mUseGravity;
        float mass = rigidbody->mMass;
        float drag = rigidbody->mDrag;
        float angularDrag = rigidbody->mAngularDrag;

        if (ImGui::Checkbox("Use Gravity", &useGravity))
        {
            Undo::recordComponent(rigidbody);

            rigidbody->mUseGravity = useGravity;
        }

        if (ImGui::InputFloat("Mass", &mass))
        {
            Undo::recordComponent(rigidbody);

            rigidbody->mMass = mass;
        }

        if (ImGui::InputFloat("Drag", &drag))
        {
            Undo::recordComponent(rigidbody);

            rigidbody->mDrag = drag;
        }

        if (ImGui::InputFloat("Angular Drag", &angularDrag))
        {
            Undo::recordComponent(rigidbody);

            rigidbody->mAngularDrag = angularDrag;
        }

        ImGui::TreePop();
    }
}