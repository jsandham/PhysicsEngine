#include "../../include/drawers/CameraDrawer.h"
#include "../../include/Undo.h"
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

void CameraDrawer::render(EditorClipboard &clipboard, Guid id)
{
    if (ImGui::TreeNodeEx("Camera", ImGuiTreeNodeFlags_DefaultOpen))
    {
        Camera *camera = clipboard.getWorld()->getComponentById<Camera>(id);
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
            Undo::recordComponent(camera);

            camera->mRenderPath = static_cast<RenderPath>(renderPath);
        }

        if (ImGui::Combo("Mode", &mode, modeNames, 2))
        {
            Undo::recordComponent(camera);

            camera->mMode = static_cast<CameraMode>(mode);
        }

        if (ImGui::Combo("SSAO", &ssao, ssaoNames, 2))
        {
            Undo::recordComponent(camera);

            camera->mSSAO = static_cast<CameraSSAO>(ssao);
        }

        glm::vec3 position = transform->mPosition; // camera->mPosition;
        glm::vec3 front = transform->getForward(); // camera->mFront;
        glm::vec3 up = transform->getUp();         // camera->mUp;
        glm::vec4 backgroundColor = glm::vec4(camera->mBackgroundColor.r, camera->mBackgroundColor.g,
                                              camera->mBackgroundColor.b, camera->mBackgroundColor.a);

        if (ImGui::ColorEdit4("Background Color", glm::value_ptr(backgroundColor)))
        {
            Undo::recordComponent(camera);

            camera->mBackgroundColor = Color(backgroundColor);
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
                Undo::recordComponent(camera);
            
                camera->setFrustum(fov, 1.0f, nearPlane, farPlane);
            }
            if (ImGui::InputFloat("Near Plane", &nearPlane))
            {
                Undo::recordComponent(camera);
               
                camera->setFrustum(fov, 1.0f, nearPlane, farPlane);
            }
            if (ImGui::InputFloat("Far Plane", &farPlane))
            {
                Undo::recordComponent(camera);
               
                camera->setFrustum(fov, 1.0f, nearPlane, farPlane);
            }

            ImGui::TreePop();
        }

        ImGui::TreePop();
    }
}