#ifndef SCENE_VIEW_H__
#define SCENE_VIEW_H__

#define GLM_FORCE_RADIANS

#include "core/World.h"
#include "systems/FreeLookCameraSystem.h"

#include "../PerformanceQueue.h"
#include "Window.h"

#include "imgui.h"
#include "ImGuizmo.h"

namespace PhysicsEditor
{
    enum class DebugTargets
    {
        Color = 0,
        ColorPicking = 1,
        Depth = 2,
        LinearDepth = 3,
        Normals = 4,
        ShadowCascades = 5,
        Position = 6,
        AlbedoSpecular = 7,
        SSAO = 8,
        SSAONoise = 9,
        Count = 10
    };

class SceneView : public Window
{
  private:
    DebugTargets mActiveDebugTarget;
    ImGuizmo::OPERATION mOperation;
    ImGuizmo::MODE mCoordinateMode;

    PerformanceQueue mPerfQueue;

    ImVec2 mSceneContentMin;
    ImVec2 mSceneContentMax;
    ImVec2 mSceneContentSize;
    bool mIsSceneContentHovered;

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

    void drawSceneHeader(Clipboard& clipboard);
    void drawSceneContent(Clipboard& clipboard);

    void drawPerformanceOverlay(Clipboard& clipboard, PhysicsEngine::FreeLookCameraSystem*cameraSystem);
    void drawCameraSettingsPopup(PhysicsEngine::FreeLookCameraSystem*cameraSystem, bool *cameraSettingsActive);
};
} // namespace PhysicsEditor

#endif
