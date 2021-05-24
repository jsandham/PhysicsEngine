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
    //if (clipboard.mSceneId.isInvalid())
    //{
    //    return;
    //}

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
    viewport.mWidth = size.x;
    viewport.mHeight = size.y;

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

    EditorCameraSystem* cameraSystem = clipboard.mEditorCameraSystem;

    cameraSystem->setViewport(viewport);

    updateWorld(clipboard.getWorld());

    const int count = 8;
    const char* textureNames[] = { "Color",    "Color Picking",   "Depth", "Normals",
                                    "Position", "Albedo/Specular", "SSAO",  "SSAO Noise" };

    const GLint textures[] = { static_cast<GLint>(cameraSystem->getNativeGraphicsColorTex()),
                                static_cast<GLint>(cameraSystem->getNativeGraphicsColorPickingTex()),
                                static_cast<GLint>(cameraSystem->getNativeGraphicsDepthTex()),
                                static_cast<GLint>(cameraSystem->getNativeGraphicsNormalTex()),
                                static_cast<GLint>(cameraSystem->getNativeGraphicsPositionTex()),
                                static_cast<GLint>(cameraSystem->getNativeGraphicsAlbedoSpecTex()),
                                static_cast<GLint>(cameraSystem->getNativeGraphicsSSAOColorTex()),
                                static_cast<GLint>(cameraSystem->getNativeGraphicsSSAONoiseTex()) };

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

    Frustum frustum;
    frustum.mNearPlane = 2.0f;
    frustum.mFarPlane = 8.0f;
    frustum.computePlanes(glm::vec3(0,0,0), glm::vec3(0,0,1), glm::vec3(0,1,0), glm::vec3(1,0,0));
    clipboard.mGizmoSystem->addToDrawList(frustum, Color(0,1,0,1), true);
    //clipboard.mGizmoSystem->addToDrawList(AABB(glm::vec3(0,0,0), glm::vec3(2,2,2)), Color(0, 1, 0, 1));

    if (clipboard.getDraggedType() == InteractionType::Mesh)
    {
        if (clipboard.mSceneViewHoveredThisFrame)
        {
            Entity* entity = clipboard.getWorld()->createEntity();
            Transform* transform = entity->addComponent<Transform>(clipboard.getWorld());
            MeshRenderer* meshRenderer = entity->addComponent<MeshRenderer>(clipboard.getWorld());
            meshRenderer->setMesh(clipboard.getDraggedId());
            meshRenderer->setMaterial(clipboard.getWorld()->getColorMaterial());

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
                float width = mSceneContentMax.x - mSceneContentMin.x;
                float height = mSceneContentMax.y - mSceneContentMin.y;

                float mousePosX = std::min(std::max(io.MousePos.x - mSceneContentMin.x, 0.0f), width);
                float mousePosY = height - std::min(std::max(io.MousePos.y - mSceneContentMin.y, 0.0f), height);

                float ndc_x = 2 * (mousePosX - 0.5f * width) / width;
                float ndc_y = 2 * (mousePosY - 0.5f * height) / height;

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

            glm::mat4 view = clipboard.mEditorCameraSystem->getViewMatrix();
            glm::mat4 projection = clipboard.mEditorCameraSystem->getProjMatrix();
            glm::mat4 model = transform->getModelMatrix();

            ImGuizmo::Manipulate(glm::value_ptr(view), glm::value_ptr(projection), operation,
                ImGuizmo::MODE::WORLD, glm::value_ptr(model), NULL, NULL);

            if (ImGuizmo::IsUsing())
            {
                glm::vec3 scale;
                glm::quat rotation;
                glm::vec3 translation;
            
                Transform::decompose(model, translation, rotation, scale);

                rotation = glm::conjugate(rotation);

                transform->mPosition = translation;
                transform->mScale = scale;
                transform->mRotation = rotation;
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

    // Mouse
    if (isFocused())
    {
        for (int i = 0; i < 5; i++)
        {
            mInput.mMouseButtonWasDown[i] = mInput.mMouseButtonIsDown[i];
            mInput.mMouseButtonIsDown[i] = false;
        }

        mInput.mMouseButtonIsDown[0] = io.MouseDown[0]; // Left Mouse Button
        mInput.mMouseButtonIsDown[1] = io.MouseDown[2]; // Middle Mouse Button
        mInput.mMouseButtonIsDown[2] = io.MouseDown[1]; // Right Mouse Button
        mInput.mMouseButtonIsDown[3] = io.MouseDown[3]; // Alt0 Mouse Button
        mInput.mMouseButtonIsDown[4] = io.MouseDown[4]; // Alt1 Mouse Button

        mInput.mMouseDelta = (int)io.MouseWheel;

        // clamp mouse position to be within the scene view content region
        ImVec2 sceneViewContentMin = getSceneContentMin();
        ImVec2 sceneViewContentMax = getSceneContentMax();

        int sceneViewContentWidth = (int)(sceneViewContentMax.x - sceneViewContentMin.x);
        int sceneViewContentHeight = (int)(sceneViewContentMax.y - sceneViewContentMin.y);

        // input->mousePosX = (int)io.MousePos.x;
        // input->mousePosY = -(int)io.MousePos.y;
        mInput.mMousePosX = std::min(std::max((int)io.MousePos.x - (int)sceneViewContentMin.x, 0), sceneViewContentWidth);
        mInput.mMousePosY =
            sceneViewContentHeight -
            std::min(std::max((int)io.MousePos.y - (int)sceneViewContentMin.y, 0), sceneViewContentHeight);
    }

    // Keyboard
    if (isFocused())
    {
        for (int i = 0; i < 61; i++)
        {
            mInput.mKeyWasDown[i] = mInput.mKeyIsDown[i];
            mInput.mKeyIsDown[i] = false;
        }

        // 0 - 9
        for (int i = 0; i < 10; i++)
        {
            mInput.mKeyIsDown[static_cast<int>(KeyCode::Key0) + i] = io.KeysDown[48 + i];
        }

        // A - Z
        for (int i = 0; i < 26; i++)
        {
            mInput.mKeyIsDown[static_cast<int>(KeyCode::A) + i] = io.KeysDown[65 + i];
        }

        mInput.mKeyIsDown[static_cast<int>(KeyCode::Enter)] = io.KeysDown[13]; // Enter
        mInput.mKeyIsDown[static_cast<int>(KeyCode::Up)] = io.KeysDown[38]; // Up
        mInput.mKeyIsDown[static_cast<int>(KeyCode::Down)] = io.KeysDown[40]; // Down
        mInput.mKeyIsDown[static_cast<int>(KeyCode::Left)] = io.KeysDown[37]; // Left
        mInput.mKeyIsDown[static_cast<int>(KeyCode::Right)] = io.KeysDown[39]; // Right
        mInput.mKeyIsDown[static_cast<int>(KeyCode::Space)] = io.KeysDown[32]; // Space
        mInput.mKeyIsDown[static_cast<int>(KeyCode::LShift)] = io.KeysDown[16]; // LShift
        mInput.mKeyIsDown[static_cast<int>(KeyCode::RShift)] = io.KeysDown[16]; // RShift
        mInput.mKeyIsDown[static_cast<int>(KeyCode::Tab)] = io.KeysDown[9];  // Tab
        mInput.mKeyIsDown[static_cast<int>(KeyCode::Backspace)] = io.KeysDown[8];  // Backspace
        mInput.mKeyIsDown[static_cast<int>(KeyCode::CapsLock)] = io.KeysDown[20]; // CapsLock
        mInput.mKeyIsDown[static_cast<int>(KeyCode::LCtrl)] = io.KeysDown[17]; // LCtrl
        mInput.mKeyIsDown[static_cast<int>(KeyCode::RCtrl)] = io.KeysDown[17]; // RCtrl
        mInput.mKeyIsDown[static_cast<int>(KeyCode::Escape)] = io.KeysDown[27]; // Escape
        mInput.mKeyIsDown[static_cast<int>(KeyCode::NumPad0)] = io.KeysDown[45]; // NumPad0
        mInput.mKeyIsDown[static_cast<int>(KeyCode::NumPad1)] = io.KeysDown[35]; // NumPad1
        mInput.mKeyIsDown[static_cast<int>(KeyCode::NumPad2)] = io.KeysDown[40]; // NumPad2
        mInput.mKeyIsDown[static_cast<int>(KeyCode::NumPad3)] = io.KeysDown[34]; // NumPad3
        mInput.mKeyIsDown[static_cast<int>(KeyCode::NumPad4)] = io.KeysDown[37]; // NumPad4
        mInput.mKeyIsDown[static_cast<int>(KeyCode::NumPad5)] = io.KeysDown[12]; // NumPad5
        mInput.mKeyIsDown[static_cast<int>(KeyCode::NumPad6)] = io.KeysDown[39]; // NumPad6
        mInput.mKeyIsDown[static_cast<int>(KeyCode::NumPad7)] = io.KeysDown[36]; // NumPad7
        mInput.mKeyIsDown[static_cast<int>(KeyCode::NumPad8)] = io.KeysDown[8];  // NumPad8
        mInput.mKeyIsDown[static_cast<int>(KeyCode::NumPad9)] = io.KeysDown[33]; // NumPad9
    }

    // call update on all systems in world
    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < world->getNumberOfUpdatingSystems(); i++)
    {
        System *system = world->getSystemByUpdateOrder(i);

        system->update(mInput, mTime);
    }
    auto end = std::chrono::steady_clock::now();

    std::chrono::duration<double> elapsed_seconds = end - start;
    mTime.deltaTime = elapsed_seconds.count();
    mTime.frameCount++;
}

void SceneView::drawPerformanceOverlay(Clipboard& clipboard, PhysicsEngine::EditorCameraSystem *cameraSystem)
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
      
        ImGui::Text("Tris: %d\n", cameraSystem->getQuery().mTris);
        ImGui::Text("Verts: %d\n", cameraSystem->getQuery().mVerts);
        ImGui::Text("Draw calls: %d\n", cameraSystem->getQuery().mNumDrawCalls);
        ImGui::Text("Elapsed time: %f", cameraSystem->getQuery().mTotalElapsedTime);
        ImGui::Text("Framerate: %f", ImGui::GetIO().Framerate);
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

void SceneView::drawCameraSettingsPopup(PhysicsEngine::EditorCameraSystem *cameraSystem, bool *cameraSettingsActive)
{
    static bool cameraSettingsWindowOpen = false;

    ImGui::SetNextWindowSize(ImVec2(430, 450), ImGuiCond_FirstUseEver);
    if (ImGui::Begin("Editor Camera Settings", cameraSettingsActive, ImGuiWindowFlags_NoResize))
    {
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
    }

    ImGui::End();
}