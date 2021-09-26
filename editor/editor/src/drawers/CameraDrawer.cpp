#include "../../include/drawers/CameraDrawer.h"
#include "../../include/Undo.h"
#include "../../include/EditorCommands.h"
#include "../../include/imgui/imgui_extensions.h"

#include "components/Camera.h"

#include "imgui.h"

using namespace PhysicsEditor;

CameraDrawer::CameraDrawer()
{
}

CameraDrawer::~CameraDrawer()
{
}

void CameraDrawer::render(Clipboard &clipboard, Guid id)
{
    InspectorDrawer::render(clipboard, id);

    ImGui::Separator();
    mContentMin = ImGui::GetItemRectMin();

    if (ImGui::TreeNodeEx("Camera", ImGuiTreeNodeFlags_DefaultOpen))
    {
        Camera *camera = clipboard.getWorld()->getComponentById<Camera>(id);

        if (camera != nullptr)
        {
            ImGui::Text(("ComponentId: " + id.toString()).c_str());

            int renderPath = static_cast<int>(camera->mRenderPath);
            int renderMode = static_cast<int>(camera->mRenderMode);
            int mode = static_cast<int>(camera->mMode);
            int ssao = static_cast<int>(camera->mSSAO);

            const char* renderPathNames[] = { "Forward", "Deferred" };
            const char* renderModeNames[] = { "Color", "Depth", "Normals" };
            const char* modeNames[] = { "Main", "Secondary" };
            const char* ssaoNames[] = { "On", "Off" };

            if (ImGui::Combo("Render Path", &renderPath, renderPathNames, 2))
            {
                camera->mRenderPath = static_cast<RenderPath>(renderPath);
            }

            if (ImGui::Combo("Render Mode", &renderMode, renderModeNames, 3))
            {
                camera->mRenderMode = static_cast<RenderMode>(renderMode);
            }

            if (ImGui::Combo("Mode", &mode, modeNames, 2))
            {
                camera->mMode = static_cast<CameraMode>(mode);
            }

            if (ImGui::Combo("SSAO", &ssao, ssaoNames, 2))
            {
                camera->mSSAO = static_cast<CameraSSAO>(ssao);
            }






            Guid renderTargetId = camera->mRenderTextureId;

            std::string renderTargetName = "None (Render Texture)";
            if (renderTargetId.isValid())
            {
                renderTargetName = renderTargetId.toString();
            }

            bool releaseTriggered = false;
            bool clearClicked = false;
            bool isClicked = ImGui::Slot("Render Target", renderTargetName, &releaseTriggered, &clearClicked);

            if (releaseTriggered && clipboard.getDraggedType() == InteractionType::RenderTexture)
            {
                renderTargetId = clipboard.getDraggedId();
                clipboard.clearDraggedItem();

                camera->mRenderTextureId = renderTargetId;
            }

            if (isClicked)
            {
                clipboard.setSelectedItem(InteractionType::RenderTexture, renderTargetId);
            }













            glm::vec4 backgroundColor = glm::vec4(camera->mBackgroundColor.r, camera->mBackgroundColor.g,
                camera->mBackgroundColor.b, camera->mBackgroundColor.a);

            if (ImGui::ColorEdit4("Background Color", glm::value_ptr(backgroundColor)))
            {
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
                    camera->setFrustum(fov, 1.0f, nearPlane, farPlane);
                }
                if (ImGui::InputFloat("Near Plane", &nearPlane))
                {
                    camera->setFrustum(fov, 1.0f, nearPlane, farPlane);
                }
                if (ImGui::InputFloat("Far Plane", &farPlane))
                {
                    camera->setFrustum(fov, 1.0f, nearPlane, farPlane);
                }

                ImGui::TreePop();
            }

            // Directional light cascade splits
            int cascadeType = static_cast<int>(camera->mShadowCascades);

            const char* cascadeTypeNames[] = { "No Cascades", "Two Cascades", "Three Cascades", "Four Cascades", "Five Cascades" };

            if (ImGui::Combo("Shadow Cascades", &cascadeType, cascadeTypeNames, 5))
            {
                camera->mShadowCascades = static_cast<ShadowCascades>(cascadeType);
            }

            if (camera->mShadowCascades != ShadowCascades::NoCascades)
            {
                ImColor colors[5] = { ImColor(1.0f, 0.0f, 0.0f),
                                      ImColor(0.0f, 1.0f, 0.0f),
                                      ImColor(0.0f, 0.0f, 1.0f),
                                      ImColor(0.0f, 1.0f, 1.0f),
                                      ImColor(0.6f, 0.0f, 0.6f) };

                std::array<int, 5> splits = camera->getCascadeSplits();
                for (size_t i = 0; i < splits.size(); i++)
                {
                    ImGui::PushItemWidth(0.125f * ImGui::GetWindowSize().x);
                    
                    ImGuiInputTextFlags flags = ImGuiInputTextFlags_None;

                    if (i <= static_cast<int>(camera->mShadowCascades))
                    {
                        ImGui::PushStyleColor(ImGuiCol_FrameBg, (ImVec4)colors[i]);
                        ImGui::PushStyleColor(ImGuiCol_FrameBgHovered, (ImVec4)colors[i]);
                        ImGui::PushStyleColor(ImGuiCol_FrameBgActive, (ImVec4)colors[i]);
                    }
                    else
                    {
                        ImGui::PushStyleColor(ImGuiCol_FrameBg, (ImVec4)ImColor(0.5f, 0.5f, 0.5f));
                        ImGui::PushStyleColor(ImGuiCol_FrameBgHovered, (ImVec4)ImColor(0.5f, 0.5f, 0.5f));
                        ImGui::PushStyleColor(ImGuiCol_FrameBgActive, (ImVec4)ImColor(0.5f, 0.5f, 0.5f));

                        flags |= ImGuiInputTextFlags_ReadOnly;
                    }

                    if (ImGui::InputInt(("##Cascade Splits" + std::to_string(i)).c_str(), &splits[i], 0, 100, flags))
                    {
                        camera->setCascadeSplit(i, splits[i]);
                    }

                    ImGui::PopStyleColor(3);
                    ImGui::PopItemWidth();
                    ImGui::SameLine();
                }
                ImGui::Text("Cascade Splits");
            }
            
            bool enabled = camera->mEnabled;
            if (ImGui::Checkbox("Enabled?", &enabled))
            {
                camera->mEnabled = enabled;
            }
        }

        ImGui::TreePop();
    }

    ImGui::Separator();
    mContentMax = ImGui::GetItemRectMax();
}