#ifndef SCENE_VIEW_H__
#define SCENE_VIEW_H__

#define GLM_FORCE_RADIANS

#include <queue>

#include "core/Input.h"
#include "core/Time.h"
#include "core/World.h"
#include "systems/FreeLookCameraSystem.h"

#include "../PerformanceQueue.h"
#include "Window.h"

#include "imgui.h"

namespace PhysicsEditor
{
class SceneView : public Window
{
  private:
    int mActiveTextureIndex;
    PerformanceQueue mPerfQueue;

    ImVec2 mSceneContentMin;
    ImVec2 mSceneContentMax;
    bool mIsSceneContentHovered;

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
    bool isSceneContentHovered() const;

  private:
    void initWorld(PhysicsEngine::World *world);
    void updateWorld(PhysicsEngine::World *world);
    void drawPerformanceOverlay(Clipboard& clipboard, PhysicsEngine::FreeLookCameraSystem*cameraSystem);
    void drawCameraSettingsPopup(PhysicsEngine::FreeLookCameraSystem*cameraSystem, bool *cameraSettingsActive);
};
} // namespace PhysicsEditor

#endif
