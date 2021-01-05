#ifndef __SCENE_VIEW_H__
#define __SCENE_VIEW_H__

#include <queue>

#include "core/Input.h"
#include "core/Time.h"
#include "core/World.h"

#include "../EditorCameraSystem.h"
#include "../PerformanceQueue.h"
#include "../TransformGizmo.h"
#include "Window.h"

#include "imgui.h"

namespace PhysicsEditor
{
class SceneView : public Window
{
  private:
    bool focused;
    bool hovered;
    int activeTextureIndex;
    PerformanceQueue perfQueue;
    TransformGizmo transformGizmo;

    ImVec2 windowPos;
    ImVec2 sceneContentMin;
    ImVec2 sceneContentMax;

    PhysicsEngine::Input input;
    PhysicsEngine::Time time;

  public:
    SceneView();
    ~SceneView();
    SceneView(const SceneView &other) = delete;
    SceneView &operator=(const SceneView &other) = delete;

    void init(EditorClipboard &clipboard);
    void update(EditorClipboard &clipboard, bool isOpenedThisFrame);

    bool isFocused() const;
    bool isHovered() const;

    ImVec2 getSceneContentMin() const;
    ImVec2 getSceneContentMax() const;
    ImVec2 getWindowPos() const;

  private:
    void initWorld(PhysicsEngine::World *world);
    void updateWorld(PhysicsEngine::World *world);
    void drawPerformanceOverlay(PhysicsEngine::EditorCameraSystem *cameraSystem);
    void drawCameraSettingsPopup(PhysicsEngine::EditorCameraSystem *cameraSystem, bool *cameraSettingsActive);
};
} // namespace PhysicsEditor

#endif
