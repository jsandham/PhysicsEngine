#include "../../include/views/SceneView.h"

#include <chrono>

#include "core/Intersect.h"
#include "core/Log.h"
#include "core/Rect.h"

#include "graphics/Graphics.h"

#include "imgui.h"
#include "ImGuizmo.h"

#include "../../include/imgui/imgui_extensions.h"

using namespace PhysicsEngine;
using namespace PhysicsEditor;

SceneView::SceneView() : Window("Scene View")
{
    mActiveTextureIndex = 0;

    mPerfQueue.setNumberOfSamples(100);

    mSceneContentMin = ImVec2(0, 0);
    mSceneContentMax = ImVec2(0, 0);
    mIsSceneContentHovered = false;

    mInput = {};
    mTime = {};
}

SceneView::~SceneView()
{
}

void SceneView::init(Clipboard &clipboard)
{
    initWorld(clipboard.getWorld());
}

void SceneView::update(Clipboard &clipboard)
{
    if (clipboard.mProjectPath.empty())
    {
        return;
    }
    /*if (clipboard.mScenePath.empty())
    {
        return;
    }*/

    static bool gizmosChecked = false;
    static bool overlayChecked = false;
    static bool cameraSettingsClicked = false;
    static bool translationModeActive = true;
    static bool rotationModeActive = false;
    static bool scaleModeActive = false;

    mSceneContentMin = getContentMin();
    mSceneContentMax = getContentMax();

    // account for the fact that Image will draw below buttons
    mSceneContentMin.y += 23;

    ImVec2 size = mSceneContentMax;
    size.x -= mSceneContentMin.x;
    size.y -= mSceneContentMin.y;

    Viewport viewport;
    viewport.mX = 0;
    viewport.mY = 0;
    viewport.mWidth = (int)size.x;
    viewport.mHeight = (int)size.y;

    ImGuiIO& io = ImGui::GetIO();
    float sceneContentWidth = (mSceneContentMax.x - mSceneContentMin.x);
    float sceneContentHeight = (mSceneContentMax.y - mSceneContentMin.y);
    float mousePosX = std::min(std::max(io.MousePos.x - mSceneContentMin.x, 0.0f), sceneContentWidth);
    float mousePosY =
        sceneContentHeight - std::min(std::max(io.MousePos.y - mSceneContentMin.y, 0.0f), sceneContentHeight);

    float nx = mousePosX / sceneContentWidth;
    float ny = mousePosY / sceneContentHeight;

    Rect sceneContentRect(mSceneContentMin.x, mSceneContentMin.y, sceneContentWidth, sceneContentHeight);

    mIsSceneContentHovered = sceneContentRect.contains(io.MousePos.x, io.MousePos.y);

    FreeLookCameraSystem* cameraSystem = clipboard.mCameraSystem;

    cameraSystem->setViewport(viewport);
    clipboard.mGizmoSystem->mEnabled = !clipboard.mScenePath.empty();

    updateWorld(clipboard.getWorld());

    const int count = 8;
    const char* textureNames[] = { "Color",    "Color Picking",   "Depth", "Normals",
                                    "Position", "Albedo/Specular", "SSAO",  "SSAO Noise" };

    const unsigned int textures[] = { cameraSystem->getNativeGraphicsColorTex(),
                                      cameraSystem->getNativeGraphicsColorPickingTex(),
                                      cameraSystem->getNativeGraphicsDepthTex(),
                                      cameraSystem->getNativeGraphicsNormalTex(),
                                      cameraSystem->getNativeGraphicsPositionTex(),
                                      cameraSystem->getNativeGraphicsAlbedoSpecTex(),
                                      cameraSystem->getNativeGraphicsSSAOColorTex(),
                                      cameraSystem->getNativeGraphicsSSAONoiseTex() };

    ImGui::PushItemWidth(0.25f * ImGui::GetWindowSize().x);

    // select draw texture dropdown
    if (ImGui::BeginCombo("##DrawTexture", textureNames[mActiveTextureIndex]))
    {
        for (int n = 0; n < count; n++)
        {
            if (textures[n] == -1)
            {
                ImGui::PushItemFlag(ImGuiItemFlags_Disabled, true);
                ImGui::PushStyleVar(ImGuiStyleVar_Alpha, ImGui::GetStyle().Alpha * 0.5f);
            }

            bool is_selected = (textureNames[mActiveTextureIndex] == textureNames[n]);
            if (ImGui::Selectable(textureNames[n], is_selected))
            {
                mActiveTextureIndex = n;

                if (is_selected)
                {
                    ImGui::SetItemDefaultFocus();
                }
            }

            if (textures[n] == -1)
            {
                ImGui::PopItemFlag();
                ImGui::PopStyleVar();
            }
        }
        ImGui::EndCombo();
    }
    ImGui::PopItemWidth();
    ImGui::SameLine();

    // whether to render gizmos or not
    if (ImGui::Checkbox("Gizmos", &gizmosChecked))
    {
        cameraSystem->setGizmos(gizmosChecked ? CameraGizmos::Gizmos_On : CameraGizmos::Gizmos_Off);
    }
    ImGui::SameLine();

    // editor rendering performance overlay
    if (ImGui::Checkbox("Perf", &overlayChecked))
    {
    }
    ImGui::SameLine();

    // select transform gizmo movement mode
    static ImGuizmo::OPERATION operation = ImGuizmo::OPERATION::TRANSLATE;
    if (ImGui::StampButton("T", translationModeActive))
    {
        translationModeActive = true;
        rotationModeActive = false;
        scaleModeActive = false;
        operation = ImGuizmo::OPERATION::TRANSLATE;
    }
    ImGui::SameLine();

    if (ImGui::StampButton("R", rotationModeActive))
    {
        translationModeActive = false;
        rotationModeActive = true;
        scaleModeActive = false;
        operation = ImGuizmo::OPERATION::ROTATE;
    }
    ImGui::SameLine();

    if (ImGui::StampButton("S", scaleModeActive))
    {
        translationModeActive = false;
        rotationModeActive = false;
        scaleModeActive = true;
        operation = ImGuizmo::OPERATION::SCALE;
    }
    ImGui::SameLine();

    std::vector<std::string> worldLocalNames = {"Local", "World"};
    ImGui::PushItemWidth(0.1f * ImGui::GetWindowSize().x);

    static int gizmoMode = static_cast<int>(ImGuizmo::MODE::LOCAL);
    if (ImGui::Combo("##world/local", &gizmoMode, worldLocalNames))
    {

    }

    ImGui::PopItemWidth();
    ImGui::SameLine();

    // editor camera settings
    if (ImGui::Button("Camera Settings"))
    {
        cameraSettingsClicked = true;
    }

    if (cameraSettingsClicked)
    {
        drawCameraSettingsPopup(cameraSystem, &cameraSettingsClicked);
    }

    // performance overlay
    if (overlayChecked)
    {
        drawPerformanceOverlay(clipboard, cameraSystem);
    }

    // Update selected entity
    if (isSceneContentHovered() && io.MouseClicked[0] && !ImGuizmo::IsOver())
    {
        Guid transformId = cameraSystem->getTransformUnderMouse(nx, ny);

        Transform* transform = clipboard.getWorld()->getComponentById<Transform>(transformId);

        if (transform != nullptr)
        {
            clipboard.setSelectedItem(InteractionType::Entity, transform->getEntityId());
        }
        else
        {
            clipboard.setSelectedItem(InteractionType::None, Guid::INVALID);
        }
    }

    clipboard.mGizmoSystem->clearDrawList();

    // draw camera gizmos
    for (int i = 0; i < clipboard.mWorld.getNumberOfComponents<Camera>(); i++)
    {
        Camera* camera = clipboard.mWorld.getComponentByIndex<Camera>(i);

        if (camera->mHide == HideFlag::None && camera->mEnabled)
        {
            Entity* entity = camera->getEntity();
            Transform* transform = clipboard.mWorld.getComponent<Transform>(entity->getId());

            glm::vec3 position = transform->mPosition;
            glm::vec3 front = transform->getForward();
            glm::vec3 up = transform->getUp();
            glm::vec3 right = transform->getRight();

            std::array<Color, 5> cascadeColors = { Color::red, Color::green, Color::blue, Color::cyan, Color::magenta};

            std::array<Frustum, 5> cascadeFrustums = camera->calcCascadeFrustums(camera->calcViewSpaceCascadeEnds());
            for (size_t j = 0; j < cascadeFrustums.size(); j++)
            {
                cascadeFrustums[j].computePlanes(position, front, up, right);
                clipboard.mGizmoSystem->addToDrawList(cascadeFrustums[j], cascadeColors[j], false);
            }
        }
    }

    if (clipboard.getDraggedType() == InteractionType::Mesh)
    {
        if (clipboard.mSceneViewHoveredThisFrame)
        {
            Entity* entity = clipboard.getWorld()->createNonPrimitive(clipboard.getDraggedId());
            Transform* transform = entity->getComponent<Transform>();
           
            clipboard.mSceneViewTempEntityId = entity->getId();
            clipboard.mSceneViewTempEntity = entity;
            clipboard.mSceneViewTempTransform = transform;
        }

        if (clipboard.mSceneViewUnhoveredThisFrame)
        {
            if (clipboard.mSceneViewTempEntityId.isValid())
            {
                clipboard.getWorld()->immediateDestroyEntity(clipboard.mSceneViewTempEntityId);
                clipboard.mSceneViewTempEntityId = Guid::INVALID;
                clipboard.mSceneViewTempEntity = nullptr;
                clipboard.mSceneViewTempTransform = nullptr;
            }
        }

        if (clipboard.mSceneViewHovered)
        {
            if (clipboard.mSceneViewTempEntityId.isValid())
            {
                float ndc_x = 2 * (mousePosX - 0.5f * sceneContentWidth) / sceneContentWidth;
                float ndc_y = 2 * (mousePosY - 0.5f * sceneContentHeight) / sceneContentHeight;

                Ray cameraRay = cameraSystem->normalizedDeviceSpaceToRay(ndc_x, ndc_y);

                Plane xz;
                xz.mX0 = glm::vec3(0, 0, 0);
                xz.mNormal = glm::vec3(0, 1, 0);

                float dist = -1.0f;
                bool intersects = Intersect::intersect(cameraRay, xz, dist);

                clipboard.mSceneViewTempTransform->mPosition = (intersects && dist >= 0.0f) ? cameraRay.getPoint(dist) : cameraRay.getPoint(5.0f);
            }
        }
    }

    // Finally draw scene
    ImGui::Image((void*)(intptr_t)textures[mActiveTextureIndex], size, ImVec2(0, size.y / 1080.0f),
        ImVec2(size.x / 1920.0f, 0));

    // draw transform gizmo if entity is selected
    if (clipboard.getSelectedType() == InteractionType::Entity)
    {
        Transform* transform = clipboard.getWorld()->getComponent<Transform>(clipboard.getSelectedId());

        if (transform != nullptr)
        {
            ImGuizmo::SetOrthographic(false);
            ImGuizmo::SetDrawlist();
            float windowWidth = ImGui::GetWindowWidth();
            float windowHeight = ImGui::GetWindowHeight();
            ImGuizmo::SetRect(ImGui::GetWindowPos().x, ImGui::GetWindowPos().y, windowWidth, windowHeight);

            glm::mat4 view = clipboard.mCameraSystem->getViewMatrix();
            glm::mat4 projection = clipboard.mCameraSystem->getProjMatrix();
            glm::mat4 model = transform->getModelMatrix();

            ImGuizmo::AllowAxisFlip(false);

            ImGuizmo::Manipulate(glm::value_ptr(view), glm::value_ptr(projection), operation,
                static_cast<ImGuizmo::MODE>(gizmoMode), glm::value_ptr(model), NULL, NULL);

            if (ImGuizmo::IsUsing())
            {
                glm::vec3 scale;
                glm::quat rotation;
                glm::vec3 translation;

                Transform::decompose(model, translation, rotation, scale);

                transform->mPosition = translation;
                transform->mScale = scale;
                transform->mRotation = rotation;
            }

            Camera* camera = clipboard.getWorld()->getComponent<Camera>(clipboard.getSelectedId());
            if (camera != nullptr && camera->mEnabled)
            {
                camera->computeViewMatrix(transform->mPosition, transform->getForward(), transform->getUp());

                ImVec2 min = mSceneContentMin;
                ImVec2 max = mSceneContentMax;

                min.x += 0.6f * getWindowWidth();
                min.y += 0.6f * getWindowHeight();

                if (camera->mRenderTextureId.isValid())
                {
                    RenderTexture* renderTexture = clipboard.getWorld()->getAssetById<RenderTexture>(camera->mRenderTextureId);
                    ImGui::GetWindowDrawList()->AddImage((void*)(intptr_t)renderTexture->getNativeGraphicsColorTex(), min, max, ImVec2(0, 1), ImVec2(1, 0));
                }
                else
                {
                    ImGui::GetWindowDrawList()->AddImage((void*)(intptr_t)camera->getNativeGraphicsColorTex(), min, max, ImVec2(0, 1), ImVec2(1, 0));
                }
            }
        }
    }
}

ImVec2 SceneView::getSceneContentMin() const
{
    return mSceneContentMin;
}

ImVec2 SceneView::getSceneContentMax() const
{
    return mSceneContentMax;
}

bool SceneView::isSceneContentHovered() const
{
    return mIsSceneContentHovered;
}

void SceneView::initWorld(PhysicsEngine::World *world)
{
    for (int i = 0; i < world->getNumberOfUpdatingSystems(); i++)
    {
        System *system = world->getSystemByUpdateOrder(i);

        system->init(world);
    }
}

void SceneView::updateWorld(World *world)
{
    ImGuiIO &io = ImGui::GetIO();

    Input input = isFocused() ? getInput() : Input();

    // Mouse
    if (isFocused())
    {
        // clamp mouse position to be within the scene view content region
        ImVec2 sceneViewContentMin = getSceneContentMin();
        ImVec2 sceneViewContentMax = getSceneContentMax();

        int sceneViewContentWidth = (int)(sceneViewContentMax.x - sceneViewContentMin.x);
        int sceneViewContentHeight = (int)(sceneViewContentMax.y - sceneViewContentMin.y);

        input.mMousePosX = std::min(std::max((int)io.MousePos.x - (int)sceneViewContentMin.x, 0), sceneViewContentWidth);
        input.mMousePosY =
            sceneViewContentHeight -
            std::min(std::max((int)io.MousePos.y - (int)sceneViewContentMin.y, 0), sceneViewContentHeight);
    }

    // call update on all systems in world
    for (int i = 0; i < world->getNumberOfUpdatingSystems(); i++)
    {
        System *system = world->getSystemByUpdateOrder(i);

        if (system->mEnabled) {
            system->update(input, mTime);
        }
    }
}

void SceneView::drawPerformanceOverlay(Clipboard& clipboard, PhysicsEngine::FreeLookCameraSystem*cameraSystem)
{
    static bool overlayOpened = false;
    static ImGuiWindowFlags overlayFlags = ImGuiWindowFlags_Tooltip | ImGuiWindowFlags_NoTitleBar |
                                           ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoSavedSettings |
                                           ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoDocking |
                                           ImGuiWindowFlags_NoNav | ImGuiWindowFlags_NoMove;

    ImVec2 overlayPos = ImVec2(mSceneContentMax.x, mSceneContentMin.y);

    ImGui::SetNextWindowPos(overlayPos, ImGuiCond_Always, ImVec2(1.0f, 0.0f));
    ImGui::SetNextWindowBgAlpha(0.35f); // Transparent background
    if (ImGui::Begin("Editor Performance Overlay", &overlayOpened, overlayFlags))
    {
        ImGuiIO& io = ImGui::GetIO();
      
        ImGui::Text("Project name: %s\n", clipboard.getProjectName().c_str());
        ImGui::Text("Project path: %s\n", clipboard.getProjectPath().string().c_str());
        ImGui::Text("Scene name: %s\n", clipboard.getSceneName().c_str());
        ImGui::Text("Scene path: %s\n", clipboard.getScenePath().string().c_str());

        ImGui::Text("Tris: %d\n", cameraSystem->getQuery().mTris);
        ImGui::Text("Verts: %d\n", cameraSystem->getQuery().mVerts);
        ImGui::Text("Draw calls: %d\n", cameraSystem->getQuery().mNumDrawCalls);
        ImGui::Text("Elapsed time: %f", cameraSystem->getQuery().mTotalElapsedTime);
        ImGui::Text("Delta time: %f", clipboard.mTime.mDeltaTime);
        ImGui::Text("ImGui::GetIO().Framerate: %f", ImGui::GetIO().Framerate);
        ImGui::Text("getFPS: %f", PhysicsEngine::getFPS(clipboard.mTime));
        ImGui::Text("Window position: %f %f\n", getWindowPos().x, getWindowPos().y);
        ImGui::Text("Scene content min: %f %f\n", mSceneContentMin.x, mSceneContentMin.y);
        ImGui::Text("Scene content max: %f %f\n", mSceneContentMax.x, mSceneContentMax.y);
        ImGui::Text("Is Scene content hovered: %d\n", mIsSceneContentHovered);
        ImGui::Text("Mouse Position: %d %d\n", cameraSystem->getMousePosX(), cameraSystem->getMousePosY());
        ImGui::Text("Mouse Position: %f %f\n", io.MousePos.x, io.MousePos.y);
        ImGui::Text("Normalized Mouse Position: %f %f\n",
                    cameraSystem->getMousePosX() / (float)(mSceneContentMax.x - mSceneContentMin.x),
                    cameraSystem->getMousePosY() / (float)(mSceneContentMax.y - mSceneContentMin.y));

        ImGui::Text("Selected interaction type %d\n", clipboard.getSelectedType());
        ImGui::Text("Selected id %s\n", clipboard.getSelectedId().toString().c_str());
        ImGui::Text("Selected path %s\n", clipboard.getSelectedPath().c_str());

        float width = (float)(mSceneContentMax.x - mSceneContentMin.x);
        float height = (float)(mSceneContentMax.y - mSceneContentMin.y);
        ImGui::Text("NDC: %f %f\n", 2 * (cameraSystem->getMousePosX() - 0.5f * width) / width,
                    2 * (cameraSystem->getMousePosY() - 0.5f * height) / height);

        ImGui::Text("Camera Position: %f %f %f\n", cameraSystem->getCameraPosition().x, cameraSystem->getCameraPosition().y, cameraSystem->getCameraPosition().z);

        ImGui::GetForegroundDrawList()->AddRect(mSceneContentMin, mSceneContentMax, 0xFFFF0000);

        mPerfQueue.addSample(cameraSystem->getQuery().mTotalElapsedTime);

        std::vector<float> perfData = mPerfQueue.getData();
        ImGui::PlotHistogram("##PerfPlot", &perfData[0], (int)perfData.size(), 0, nullptr, 0, 1.0f);
    }
    ImGui::End();
}

void SceneView::drawCameraSettingsPopup(PhysicsEngine::FreeLookCameraSystem*cameraSystem, bool *cameraSettingsActive)
{
    static bool cameraSettingsWindowOpen = false;

    ImGui::SetNextWindowSize(ImVec2(430, 450), ImGuiCond_FirstUseEver);
    if (ImGui::Begin("Editor Camera Settings", cameraSettingsActive, ImGuiWindowFlags_NoResize))
    {
        // Editor camera transform
        Transform* transform = cameraSystem->getCamera()->getComponent<Transform>();
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

        // Viewport viewport = cameraSystem->getViewport();
        Frustum frustum = cameraSystem->getFrustum();

        // Viewport settings
        /*if (ImGui::InputInt("X", &viewport.mX)) {
            cameraSystem->setViewport(viewport);
        }
        if (ImGui::InputInt("Y", &viewport.mY)) {
            cameraSystem->setViewport(viewport);
        }
        if (ImGui::InputInt("Width", &viewport.mWidth)) {
            cameraSystem->setViewport(viewport);
        }
        if (ImGui::InputInt("Height", &viewport.mHeight)) {
            cameraSystem->setViewport(viewport);
        }*/

        // Frustum settings
        if (ImGui::InputFloat("FOV", &frustum.mFov))
        {
            cameraSystem->setFrustum(frustum);
        }
        if (ImGui::InputFloat("Aspect Ratio", &frustum.mAspectRatio))
        {
            cameraSystem->setFrustum(frustum);
        }
        if (ImGui::InputFloat("Near Plane", &frustum.mNearPlane))
        {
            cameraSystem->setFrustum(frustum);
        }
        if (ImGui::InputFloat("Far Plane", &frustum.mFarPlane))
        {
            cameraSystem->setFrustum(frustum);
        }

        // SSAO and render path
        int renderPath = static_cast<int>(cameraSystem->getRenderPath());
        int ssao = static_cast<int>(cameraSystem->getSSAO());

        const char *renderPathNames[] = {"Forward", "Deferred"};
        const char *ssaoNames[] = {"On", "Off"};

        if (ImGui::Combo("Render Path", &renderPath, renderPathNames, 2))
        {
            cameraSystem->setRenderPath(static_cast<RenderPath>(renderPath));
        }

        if (ImGui::Combo("SSAO", &ssao, ssaoNames, 2))
        {
            cameraSystem->setSSAO(static_cast<CameraSSAO>(ssao));
        }

        // Directional light cascade splits
        int cascadeType = static_cast<int>(cameraSystem->getCamera()->mShadowCascades);

        const char* cascadeTypeNames[] = { "No Cascades", "Two Cascades", "Three Cascades", "Four Cascades", "Five Cascades" };

        if (ImGui::Combo("Shadow Cascades", &cascadeType, cascadeTypeNames, 5))
        {
            cameraSystem->getCamera()->mShadowCascades = static_cast<ShadowCascades>(cascadeType);
        }

        if (cameraSystem->getCamera()->mShadowCascades != ShadowCascades::NoCascades)
        {
            ImColor colors[5] = { ImColor(1.0f, 0.0f, 0.0f),
                              ImColor(0.0f, 1.0f, 0.0f),
                              ImColor(0.0f, 0.0f, 1.0f),
                              ImColor(0.0f, 1.0f, 1.0f),
                              ImColor(0.6f, 0.0f, 0.6f) };

            std::array<int, 5> splits = cameraSystem->getCamera()->getCascadeSplits();
            for (size_t i = 0; i < splits.size(); i++)
            {
                ImGui::PushItemWidth(0.125f * ImGui::GetWindowSize().x);

                ImGuiInputTextFlags flags = ImGuiInputTextFlags_None;

                if (i <= static_cast<int>(cameraSystem->getCamera()->mShadowCascades))
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
                    cameraSystem->getCamera()->setCascadeSplit(i, splits[i]);
                }

                ImGui::PopStyleColor(3);
                ImGui::PopItemWidth();
                ImGui::SameLine();
            }
            ImGui::Text("Cascade Splits");
        }
    }

    ImGui::End();
}