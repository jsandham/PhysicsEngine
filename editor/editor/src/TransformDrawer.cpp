#include <math.h>

#include "../include/CommandManager.h"
#include "../include/EditorCommands.h"
#include "../include/TransformDrawer.h"

#include "components/Transform.h"

#include "imgui.h"
#include "imgui_impl_opengl3.h"
#include "imgui_impl_win32.h"
#include "imgui_internal.h"

using namespace PhysicsEditor;

TransformDrawer::TransformDrawer()
{
}

TransformDrawer::~TransformDrawer()
{
}

void TransformDrawer::render(World *world, EditorProject &project, EditorScene &scene, EditorClipboard &clipboard,
                             Guid id)
{
    if (ImGui::TreeNodeEx("Transform", ImGuiTreeNodeFlags_DefaultOpen))
    {
        Transform *transform = world->getComponentById<Transform>(id);

        ImGui::Text(("EntityId: " + transform->getEntityId().toString()).c_str());
        ImGui::Text(("ComponentId: " + id.toString()).c_str());
        ImGui::Text(("ComponentId: " + transform->getId().toString()).c_str());

        glm::vec3 position = transform->mPosition;
        glm::quat rotation = transform->mRotation;
        glm::vec3 scale = transform->mScale;
        glm::vec3 eulerAngles = glm::degrees(glm::eulerAngles(rotation));

        if (ImGui::InputFloat3("Position", glm::value_ptr(position)))
        {
            // transform->mPosition = position;
            CommandManager::addCommand(
                new ChangePropertyCommand<glm::vec3>(&transform->mPosition, position, &scene.isDirty));
        }
        /*if (ImGui::InputFloat4("Rotation", glm::value_ptr(rotation)))
        {
            CommandManager::addCommand(
                new ChangePropertyCommand<glm::quat>(&transform->mRotation, rotation, &scene.isDirty));
        }*/
        if (ImGui::InputFloat3("Rotation", glm::value_ptr(eulerAngles)))
        {
            glm::quat x = glm::angleAxis(glm::radians(eulerAngles.x), glm::vec3(1.0f, 0.0f, 0.0f));
            glm::quat y = glm::angleAxis(glm::radians(eulerAngles.y), glm::vec3(0.0f, 1.0f, 0.0f));
            glm::quat z = glm::angleAxis(glm::radians(eulerAngles.z), glm::vec3(0.0f, 0.0f, 1.0f));

            transform->mRotation = z * y * x;
        }
        if (ImGui::InputFloat3("Scale", glm::value_ptr(scale)))
        {
            CommandManager::addCommand(new ChangePropertyCommand<glm::vec3>(&transform->mScale, scale, &scene.isDirty));
        }

        ImGui::TreePop();
    }
}