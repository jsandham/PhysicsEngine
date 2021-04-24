#include "../../include/views/SceneView.h"

#include <chrono>

#include "core/Intersect.h"
#include "core/Log.h"

#include "graphics/Graphics.h"

#include "imgui.h"

#include "../../include/imgui/imgui_extensions.h"

using namespace PhysicsEngine;
using namespace PhysicsEditor;

SceneView::SceneView() : Window("Scene View")
{
    mActiveTextureIndex = 0;

    mPerfQueue.setNumberOfSamples(100);

    mSceneContentMin = ImVec2(0, 0);
    mSceneContentMax = ImVec2(0, 0);

    mTransformGizmo.initialize();

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

    EditorCameraSystem* cameraSystem = clipboard.getWorld()->getSystem<EditorCameraSystem>();

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
    if (ImGui::StampButton("T", translationModeActive))
    {
        translationModeActive = true;
        rotationModeActive = false;
        scaleModeActive = false;

        mTransformGizmo.setGizmoMode(GizmoMode::Translation);
    }
    ImGui::SameLine();

    if (ImGui::StampButton("R", rotationModeActive))
    {
        translationModeActive = false;
        rotationModeActive = true;
        scaleModeActive = false;

        mTransformGizmo.setGizmoMode(GizmoMode::Rotation);
    }
    ImGui::SameLine();

    if (ImGui::StampButton("S", scaleModeActive))
    {
        translationModeActive = false;
        rotationModeActive = false;
        scaleModeActive = true;

        mTransformGizmo.setGizmoMode(GizmoMode::Scale);
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

    ImGuiIO& io = ImGui::GetIO();
    float sceneContentWidth = (mSceneContentMax.x - mSceneContentMin.x);
    float sceneContentHeight = (mSceneContentMax.y - mSceneContentMin.y);
    float mousePosX = std::min(std::max(io.MousePos.x - mSceneContentMin.x, 0.0f), sceneContentWidth);
    float mousePosY =
        sceneContentHeight - std::min(std::max(io.MousePos.y - mSceneContentMin.y, 0.0f), sceneContentHeight);

    float nx = mousePosX / sceneContentWidth;
    float ny = mousePosY / sceneContentHeight;

    // Update selected entity
    if (isHovered() && io.MouseClicked[0] && !mTransformGizmo.isGizmoHighlighted())
    {
        Guid transformId = cameraSystem->getTransformUnderMouse(nx, ny);

        Transform* transform = clipboard.getWorld()->getComponentById<Transform>(transformId);

        if (transform != NULL)
        {
            clipboard.setSelectedItem(InteractionType::Entity, transform->getEntityId());
        }
        else
        {
            clipboard.setSelectedItem(InteractionType::None, Guid::INVALID);
        }
    }

    GizmoSystem* gizmoSystem = clipboard.getWorld()->getSystem<GizmoSystem>();

    gizmoSystem->clearDrawList();

    // draw transform gizmo if entity is selected
    if (clipboard.getSelectedType() == InteractionType::Entity)
    {
        Transform* transform = clipboard.getWorld()->getComponent<Transform>(clipboard.getSelectedId());

        if(transform != nullptr)
        {
            mTransformGizmo.update(cameraSystem, gizmoSystem, transform, mousePosX, mousePosY, sceneContentWidth,
                sceneContentHeight);
        }
    }

    // Finally draw scene
    ImGui::Image((void*)(intptr_t)textures[mActiveTextureIndex], size, ImVec2(0, size.y / 1080.0f),
        ImVec2(size.x / 1920.0f, 0));
}

ImVec2 SceneView::getSceneContentMin() const
{
    return mSceneContentMin;
}

ImVec2 SceneView::getSceneContentMax() const
{
    return mSceneContentMax;
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
            mInput.mouseButtonWasDown[i] = mInput.mouseButtonIsDown[i];
            mInput.mouseButtonIsDown[i] = false;
        }

        mInput.mouseButtonIsDown[0] = io.MouseDown[0]; // Left Mouse Button
        mInput.mouseButtonIsDown[1] = io.MouseDown[2]; // Middle Mouse Button
        mInput.mouseButtonIsDown[2] = io.MouseDown[1]; // Right Mouse Button
        mInput.mouseButtonIsDown[3] = io.MouseDown[3]; // Alt0 Mouse Button
        mInput.mouseButtonIsDown[4] = io.MouseDown[4]; // Alt1 Mouse Button

        mInput.mouseDelta = (int)io.MouseWheel;

        // clamp mouse position to be within the scene view content region
        ImVec2 sceneViewContentMin = getSceneContentMin();
        ImVec2 sceneViewContentMax = getSceneContentMax();

        int sceneViewContentWidth = (int)(sceneViewContentMax.x - sceneViewContentMin.x);
        int sceneViewContentHeight = (int)(sceneViewContentMax.y - sceneViewContentMin.y);

        // input->mousePosX = (int)io.MousePos.x;
        // input->mousePosY = -(int)io.MousePos.y;
        mInput.mousePosX = std::min(std::max((int)io.MousePos.x - (int)sceneViewContentMin.x, 0), sceneViewContentWidth);
        mInput.mousePosY =
            sceneViewContentHeight -
            std::min(std::max((int)io.MousePos.y - (int)sceneViewContentMin.y, 0), sceneViewContentHeight);
    }

    // Keyboard
    if (isFocused())
    {
        for (int i = 0; i < 61; i++)
        {
            mInput.keyWasDown[i] = mInput.keyIsDown[i];
            mInput.keyIsDown[i] = false;
        }

        // 0 - 9
        for (int i = 0; i < 10; i++)
        {
            mInput.keyIsDown[0] = io.KeysDown[48 + i];
        }

        // A - Z
        for (int i = 0; i < 26; i++)
        {
            mInput.keyIsDown[10 + i] = io.KeysDown[65 + i];
        }

        mInput.keyIsDown[36] = io.KeysDown[13]; // Enter
        mInput.keyIsDown[37] = io.KeysDown[38]; // Up
        mInput.keyIsDown[38] = io.KeysDown[40]; // Down
        mInput.keyIsDown[39] = io.KeysDown[37]; // Left
        mInput.keyIsDown[40] = io.KeysDown[39]; // Right
        mInput.keyIsDown[41] = io.KeysDown[32]; // Space
        mInput.keyIsDown[42] = io.KeysDown[16]; // LShift
        mInput.keyIsDown[43] = io.KeysDown[16]; // RShift
        mInput.keyIsDown[44] = io.KeysDown[9];  // Tab
        mInput.keyIsDown[45] = io.KeysDown[8];  // Backspace
        mInput.keyIsDown[46] = io.KeysDown[20]; // CapsLock
        mInput.keyIsDown[47] = io.KeysDown[17]; // LCtrl
        mInput.keyIsDown[48] = io.KeysDown[17]; // RCtrl
        mInput.keyIsDown[49] = io.KeysDown[27]; // Escape
        mInput.keyIsDown[50] = io.KeysDown[45]; // NumPad0
        mInput.keyIsDown[51] = io.KeysDown[35]; // NumPad1
        mInput.keyIsDown[52] = io.KeysDown[40]; // NumPad2
        mInput.keyIsDown[53] = io.KeysDown[34]; // NumPad3
        mInput.keyIsDown[54] = io.KeysDown[37]; // NumPad4
        mInput.keyIsDown[55] = io.KeysDown[12]; // NumPad5
        mInput.keyIsDown[56] = io.KeysDown[39]; // NumPad6
        mInput.keyIsDown[57] = io.KeysDown[36]; // NumPad7
        mInput.keyIsDown[58] = io.KeysDown[8];  // NumPad8
        mInput.keyIsDown[59] = io.KeysDown[33]; // NumPad9
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
        ImGui::Text("Tris: %d\n", cameraSystem->getQuery().mTris);
        ImGui::Text("Verts: %d\n", cameraSystem->getQuery().mVerts);
        ImGui::Text("Draw calls: %d\n", cameraSystem->getQuery().mNumDrawCalls);
        ImGui::Text("Elapsed time: %f", cameraSystem->getQuery().mTotalElapsedTime);
        ImGui::Text("Window position: %f %f\n", getWindowPos().x, getWindowPos().y);
        // ImGui::Text("Window position: %f %f\n", windowPos.x, windowPos.y);
        // ImGui::Text("Content min: %f %f\n", contentMin.x, contentMin.y);
        // ImGui::Text("Content max: %f %f\n", contentMax.x, contentMax.y);
        ImGui::Text("Scene content min: %f %f\n", mSceneContentMin.x, mSceneContentMin.y);
        ImGui::Text("Scene content max: %f %f\n", mSceneContentMax.x, mSceneContentMax.y);
        ImGui::Text("Mouse Position: %d %d\n", cameraSystem->getMousePosX(), cameraSystem->getMousePosY());
        ImGui::Text("Normalized Mouse Position: %f %f\n",
                    cameraSystem->getMousePosX() / (float)(mSceneContentMax.x - mSceneContentMin.x),
                    cameraSystem->getMousePosY() / (float)(mSceneContentMax.y - mSceneContentMin.y));

        ImGui::Text("Is SceneView hovered? %d\n", clipboard.mSceneViewHovered);
        ImGui::Text("Is Inspector hovered? %d\n", clipboard.mInspectorHovered);
        ImGui::Text("Is Hierarchy hovered? %d\n", clipboard.mHierarchyHovered);
        ImGui::Text("Is Console hovered? %d\n", clipboard.mConsoleHovered);
        ImGui::Text("Is ProjectView hovered? %d\n", clipboard.mProjectViewHovered);

        ImGui::Text("Selected interaction type %d\n", clipboard.getSelectedType());
        ImGui::Text("Selected id %s\n", clipboard.getSelectedId().toString().c_str());
        ImGui::Text("Selected path %s\n", clipboard.getSelectedPath().c_str());

        float width = (float)(mSceneContentMax.x - mSceneContentMin.x);
        float height = (float)(mSceneContentMax.y - mSceneContentMin.y);
        ImGui::Text("NDC: %f %f\n", 2 * (cameraSystem->getMousePosX() - 0.5f * width) / width,
                    2 * (cameraSystem->getMousePosY() - 0.5f * height) / height);

        ImGui::GetForegroundDrawList()->AddRect(mSceneContentMin, mSceneContentMax, 0xFFFF0000);

        mPerfQueue.addSample(cameraSystem->getQuery().mTotalElapsedTime);

        std::vector<float> perfData = mPerfQueue.getData();
        ImGui::PlotHistogram("##PerfPlot", &perfData[0], (int)perfData.size(), 0, nullptr, 0, 1.0f);
        // ImGui::PlotLines("Curve", &perfData[0], perfData.size());

        ImGui::Text("Active project name: %s\n", clipboard.getProjectName().c_str());
        ImGui::Text("Active project path: %s\n", clipboard.getProjectPath().c_str());
        ImGui::Text("Active scene name: %s\n", clipboard.getSceneName().c_str());
        ImGui::Text("Active scene path: %s\n", clipboard.getScenePath().c_str());
        ImGui::Text("Active scene id: %s\n", clipboard.getSceneId().toString().c_str());
        ImGui::Text("Scene count in world: %d\n", clipboard.getWorld()->getNumberOfScenes());
        ImGui::Text("Entity count in world: %d\n", clipboard.getWorld()->getNumberOfEntities());
        ImGui::Text("Transform count in world: %d\n", clipboard.getWorld()->getNumberOfComponents<Transform>());
        ImGui::Text("MeshRenderer count in world: %d\n", clipboard.getWorld()->getNumberOfComponents<MeshRenderer>());
        ImGui::Text("Light count in world: %d\n", clipboard.getWorld()->getNumberOfComponents<Light>());
        ImGui::Text("Camera count in world: %d\n", clipboard.getWorld()->getNumberOfComponents<Camera>());
        ImGui::Text("Mesh count in world: %d\n", clipboard.getWorld()->getNumberOfAssets<Mesh>());
        ImGui::Text("Material count in world: %d\n", clipboard.getWorld()->getNumberOfAssets<Material>());
        ImGui::Text("Texture2D count in world: %d\n", clipboard.getWorld()->getNumberOfAssets<Texture2D>());
        ImGui::Text("RenderSystem count in world: %d\n", clipboard.getWorld()->getNumberOfSystems<RenderSystem>());
        ImGui::Text("PhysicsSystem count in world: %d\n", clipboard.getWorld()->getNumberOfSystems<PhysicsSystem>());
        ImGui::Text("CleanUpSystem count in world: %d\n", clipboard.getWorld()->getNumberOfSystems<CleanUpSystem>());
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