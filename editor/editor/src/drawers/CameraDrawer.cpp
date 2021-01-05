#include "../../include/drawers/CameraDrawer.h"
#include "../../include/CommandManager.h"
#include "../../include/EditorCommands.h"

#include "components/Camera.h"

#include "imgui.h"
#include "imgui_impl_opengl3.h"
#include "imgui_impl_win32.h"
#include "imgui_internal.h"

using namespace PhysicsEditor;

CameraDrawer::CameraDrawer()
{
}

CameraDrawer::~CameraDrawer()
{
}

void CameraDrawer::render(EditorClipboard& clipboard, Guid id)
{
    if (ImGui::TreeNodeEx("Camera", ImGuiTreeNodeFlags_DefaultOpen))
    {
        Camera* camera = clipboard.getWorld()->getComponentById<Camera>(id);
        Transform *transform = camera->getComponent<Transform>(clipboard.getWorld());

        ImGui::Text(("EntityId: " + camera->getEntityId().toString()).c_str());
        ImGui::Text(("ComponentId: " + id.toString()).c_str());

        int renderPath = static_cast<int>(camera->mRenderPath);
        int mode = static_cast<int>(camera->mMode);
        int ssao = static_cast<int>(camera->mSSAO);

        const char *renderPathNames[] = {"Forward", "Deferred"};
        const char *modeNames[] = {"Main", "Secondary"};
        const char *ssaoNames[] = {"On", "Off"};

        if (ImGui::Combo("Render Path", &renderPath, renderPathNames, 2))
        {
            CommandManager::addCommand(new ChangePropertyCommand<RenderPath>(
                &camera->mRenderPath, static_cast<RenderPath>(renderPath), &clipboard.isDirty));
        }

        if (ImGui::Combo("Mode", &mode, modeNames, 2))
        {
            CommandManager::addCommand(
                new ChangePropertyCommand<CameraMode>(&camera->mMode, static_cast<CameraMode>(mode), &clipboard.isDirty));
        }

        if (ImGui::Combo("SSAO", &ssao, ssaoNames, 2))
        {
            CommandManager::addCommand(
                new ChangePropertyCommand<CameraSSAO>(&camera->mSSAO, static_cast<CameraSSAO>(ssao), &clipboard.isDirty));
        }

        glm::vec3 position = transform->mPosition; // camera->mPosition;
        glm::vec3 front = transform->getForward(); // camera->mFront;
        glm::vec3 up = transform->getUp();         // camera->mUp;
        glm::vec4 backgroundColor = glm::vec4(camera->mBackgroundColor.r, camera->mBackgroundColor.g,
                                              camera->mBackgroundColor.b, camera->mBackgroundColor.a);

        if (ImGui::InputFloat3("Position", glm::value_ptr(position)))
        {
            transform->mPosition = position;
            // CommandManager::addCommand(new ChangePropertyCommand<glm::vec3>(&camera->mPosition, position,
            // &scene.isDirty));
        }
        if (ImGui::InputFloat3("Front", glm::value_ptr(front)))
        {
            // CommandManager::addCommand(new ChangePropertyCommand<glm::vec3>(&camera->mFront, front, &scene.isDirty));
        }
        if (ImGui::InputFloat3("Up", glm::value_ptr(up)))
        {
            // CommandManager::addCommand(new ChangePropertyCommand<glm::vec3>(&camera->mUp, up, &scene.isDirty));
        }
        if (ImGui::ColorEdit4("Background Color", glm::value_ptr(backgroundColor)))
        {
            CommandManager::addCommand(
                new ChangePropertyCommand<Color>(&camera->mBackgroundColor, Color(backgroundColor), &clipboard.isDirty));
        }

        if (ImGui::TreeNode("Viewport"))
        {
            int x = camera->getViewport().mX;
            int y = camera->getViewport().mY;
            int width = camera->getViewport().mWidth;
            int height = camera->getViewport().mHeight;

            if (ImGui::InputInt("x", &x))
            {
                // CommandManager::addCommand(new ChangePropertyCommand<int>(&camera->mViewport.mX, x, &scene.isDirty));
            }
            if (ImGui::InputInt("y", &y))
            {
                // CommandManager::addCommand(new ChangePropertyCommand<int>(&camera->mViewport.mY, y, &scene.isDirty));
            }
            if (ImGui::InputInt("Width", &width))
            {
                // CommandManager::addCommand(new ChangePropertyCommand<int>(&camera->mViewport.mWidth, width,
                // &scene.isDirty));
            }
            if (ImGui::InputInt("Height", &height))
            {
                // CommandManager::addCommand(new ChangePropertyCommand<int>(&camera->mViewport.mHeight, height,
                // &scene.isDirty));
            }

            camera->setViewport(x, y, width, height);

            ImGui::TreePop();
        }

        if (ImGui::TreeNode("Frustum"))
        {
            float fov = camera->getFrustum().mFov;
            float nearPlane = camera->getFrustum().mNearPlane;
            float farPlane = camera->getFrustum().mFarPlane;

            if (ImGui::InputFloat("Field of View", &fov))
            {
                /*CommandManager::addCommand(
                    new ChangePropertyCommand<float>(&camera->mFrustum.mFov, fov, &scene.isDirty));*/
                camera->setFrustum(fov, 1.0f, nearPlane, farPlane);
            }
            if (ImGui::InputFloat("Near Plane", &nearPlane))
            {
                //CommandManager::addCommand(
                //    new ChangePropertyCommand<float>(&camera->mFrustum.mNearPlane, nearPlane, &scene.isDirty));
                camera->setFrustum(fov, 1.0f, nearPlane, farPlane);
            }
            if (ImGui::InputFloat("Far Plane", &farPlane))
            {
                //CommandManager::addCommand(
                //    new ChangePropertyCommand<float>(&camera->mFrustum.mFarPlane, farPlane, &scene.isDirty));
                camera->setFrustum(fov, 1.0f, nearPlane, farPlane);
            }

            ImGui::TreePop();
        }

        ImGui::TreePop();
    }
}