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

void RigidbodyDrawer::render(Clipboard &clipboard, const PhysicsEngine::Guid& id)
{
    ImGui::Separator();
    mContentMin = ImGui::GetItemRectMin();

    if (ImGui::TreeNodeEx("Rigidbody", ImGuiTreeNodeFlags_DefaultOpen))
    {
        PhysicsEngine::Rigidbody *rigidbody = clipboard.getWorld()->getActiveScene()->getComponentByGuid<PhysicsEngine::Rigidbody>(id);

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

    if (isHovered())
    {
        if (ImGui::BeginPopupContextWindow("RightMouseClickPopup"))
        {
            if (ImGui::MenuItem("RemoveComponent", NULL, false, true))
            {
                PhysicsEngine::Rigidbody* rigidbody = clipboard.getWorld()->getActiveScene()->getComponentByGuid<PhysicsEngine::Rigidbody>(id);
                clipboard.getWorld()->getActiveScene()->immediateDestroyComponent(rigidbody->getEntityGuid(), id, PhysicsEngine::ComponentType<PhysicsEngine::Rigidbody>::type);
            }

            ImGui::EndPopup();
        }
    }
}

bool RigidbodyDrawer::isHovered() const
{
    ImVec2 cursorPos = ImGui::GetMousePos();

    glm::vec2 min = glm::vec2(mContentMin.x, mContentMin.y);
    glm::vec2 max = glm::vec2(mContentMax.x, mContentMax.y);

    PhysicsEngine::Rect rect(min, max);

    return rect.contains(cursorPos.x, cursorPos.y);
}