#include "../../include/drawers/TransformDrawer.h"

#include <math.h>

#include "../../include/EditorClipboard.h"

#include "components/Transform.h"

#include "imgui.h"

using namespace PhysicsEditor;

TransformDrawer::TransformDrawer()
{
}

TransformDrawer::~TransformDrawer()
{
}

void TransformDrawer::render(Clipboard &clipboard, const Guid& id)
{
    InspectorDrawer::render(clipboard, id);

    ImGui::Separator();
    mContentMin = ImGui::GetItemRectMin();

    if (ImGui::TreeNodeEx("Transform", ImGuiTreeNodeFlags_DefaultOpen))
    {
        Transform *transform = clipboard.getWorld()->getComponentById<Transform>(id);

        if (transform != nullptr)
        {
            ImGui::Text(("ComponentId: " + transform->getId().toString()).c_str());

            glm::vec3 position = transform->mPosition;
            glm::quat rotation = transform->mRotation;
            glm::vec3 scale = transform->mScale;
            glm::vec3 eulerAngles = glm::degrees(glm::eulerAngles(rotation));

            if (ImGui::InputFloat3("Position", glm::value_ptr(position)))
            {
                transform->mPosition = position;
            }

            if (ImGui::InputFloat3("Rotation", glm::value_ptr(eulerAngles)))
            {
                glm::quat x = glm::angleAxis(glm::radians(eulerAngles.x), glm::vec3(1.0f, 0.0f, 0.0f));
                glm::quat y = glm::angleAxis(glm::radians(eulerAngles.y), glm::vec3(0.0f, 1.0f, 0.0f));
                glm::quat z = glm::angleAxis(glm::radians(eulerAngles.z), glm::vec3(0.0f, 0.0f, 1.0f));

                transform->mRotation = z * y * x;
            }
            if (ImGui::InputFloat3("Scale", glm::value_ptr(scale)))
            {
                transform->mScale = scale;
            }
        }

        ImGui::TreePop();
    }

    ImGui::Separator();
    mContentMax = ImGui::GetItemRectMax();
}