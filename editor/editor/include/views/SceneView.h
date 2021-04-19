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
    int mActiveTextureIndex;
    PerformanceQueue mPerfQueue;
    TransformGizmo mTransformGizmo;

    ImVec2 mSceneContentMin;
    ImVec2 mSceneContentMax;

    PhysicsEngine::Input mInput;
    PhysicsEngine::Time mTime;

  public:
    SceneView();
    ~SceneView();
    SceneView(const SceneView &other) = delete;
    SceneView &operator=(const SceneView &other) = delete;

    void init(Clipboard &clipboard) override;
    void update(Clipboard &clipboard) override;

    ImVec2 getSceneContentMin() const;
    ImVec2 getSceneContentMax() const;

  private:
    void initWorld(PhysicsEngine::World *world);
    void updateWorld(PhysicsEngine::World *world);
    void drawPerformanceOverlay(Clipboard& clipboard, PhysicsEngine::EditorCameraSystem *cameraSystem);
    void drawCameraSettingsPopup(PhysicsEngine::EditorCameraSystem *cameraSystem, bool *cameraSettingsActive);
};
} // namespace PhysicsEditor

#endif
