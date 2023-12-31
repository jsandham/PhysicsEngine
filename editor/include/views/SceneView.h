#ifndef SCENE_VIEW_H__
#define SCENE_VIEW_H__

#include "core/World.h"
#include "systems/FreeLookCameraSystem.h"

#include "../EditorClipboard.h"

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
		OcclusionMap = 10,
		Raytracing = 11,
		Count = 12
	};

	class SceneView
	{
	private:
		DebugTargets mActiveDebugTarget;
		ImGuizmo::OPERATION mOperation;
		ImGuizmo::MODE mCoordinateMode;

		ImVec2 mWindowPos;
		ImVec2 mContentMin;
		ImVec2 mContentMax;
		float mWindowWidth;
		float mWindowHeight;

		bool mOpen;
		bool mFocused;
		bool mHovered;
		bool mHoveredLastFrame;

		ImVec2 mSceneContentMin;
		ImVec2 mSceneContentMax;
		ImVec2 mSceneContentSize;
		bool mIsSceneContentHovered;

	public:
		SceneView();
		~SceneView();
		SceneView(const SceneView& other) = delete;
		SceneView& operator=(const SceneView& other) = delete;

		void init(Clipboard& clipboard);
		void update(Clipboard& clipboard, bool isOpenedThisFrame);

		ImVec2 getWindowPos() const;
		ImVec2 getContentMin() const;
		ImVec2 getContentMax() const;

		float getWindowWidth() const;
		float getWindowHeight() const;

		bool isOpen() const;
		bool isFocused() const;
		bool isHovered() const;
		bool hoveredThisFrame() const;
		bool unhoveredThisFrame() const;

		ImVec2 getSceneContentMin() const;
		ImVec2 getSceneContentMax() const;
		bool isSceneContentHovered() const;

	private:
		void initWorld(PhysicsEngine::World* world);
		void updateWorld(PhysicsEngine::World* world);

		void drawSceneHeader(Clipboard& clipboard);
		void drawSceneContent(Clipboard& clipboard);

		void drawPerformanceOverlay(Clipboard& clipboard, PhysicsEngine::FreeLookCameraSystem* cameraSystem);
		void drawCameraSettingsPopup(PhysicsEngine::FreeLookCameraSystem* cameraSystem, bool* cameraSettingsActive);
	};
} // namespace PhysicsEditor

#endif
