#include "../include/SceneView.h"

#include "core/Log.h"

#include "imgui_impl_win32.h"
#include "imgui_impl_opengl3.h"
#include "imgui_internal.h"
#include "imgui.h"

#include "../include/imgui_extensions.h"

#include "core/ClosestDistance.h"

using namespace PhysicsEngine;
using namespace PhysicsEditor;

SceneView::SceneView()
{
	focused = false;
	hovered = false;

	activeTextureIndex = 0;

	perfQueue.setNumberOfSamples(100);

	windowPos = ImVec2(0, 0);
	sceneContentMin = ImVec2(0, 0);
	sceneContentMax = ImVec2(0, 0);

	transformGizmo.initialize();
}

SceneView::~SceneView()
{

}

void SceneView::render(PhysicsEngine::World* world, 
					   PhysicsEngine::EditorCameraSystem* cameraSystem, 
					   EditorClipboard& clipboard, 
					   bool isOpenedThisFrame)
{
	focused = false;
	hovered = false;
	static bool sceneViewActive = true;

	if (isOpenedThisFrame) {
		sceneViewActive = true;
	}

	if (!sceneViewActive) {
		return;
	}

	static bool gizmosChecked = false;
	static bool overlayChecked = false;
	static bool cameraSettingsClicked = false;
	static bool translationModeActive = true;
	static bool rotationModeActive = false;
	static bool scaleModeActive = false;

	ImGui::Begin("Scene View", &sceneViewActive);
	{
		if (ImGui::GetIO().MouseClicked[1] && ImGui::IsWindowHovered()) {
			ImGui::SetWindowFocus("Scene View");
		}

		focused = ImGui::IsWindowFocused();
		hovered = ImGui::IsWindowHovered();
		windowPos = ImGui::GetWindowPos();
		
		ImVec2 contentMin = ImGui::GetWindowContentRegionMin();
		ImVec2 contentMax = ImGui::GetWindowContentRegionMax();

		contentMin.x += windowPos.x;
		contentMin.y += windowPos.y;
		contentMax.x += windowPos.x;
		contentMax.y += windowPos.y;

		sceneContentMin = contentMin;
		sceneContentMax = contentMax;

		// account for the fact that Image will draw below buttons
		sceneContentMin.y += 23;

		const int count = 8;
		const char* textureNames[] = { "Color",
									   "Color Picking",
									   "Depth",
									   "Normals",
									   "Position",
									   "Albedo/Specular",
									   "SSAO",
									   "SSAO Noise" };

		const GLint textures[] = { static_cast<GLint>(cameraSystem->getNativeGraphicsColorTex()),
								   static_cast<GLint>(cameraSystem->getNativeGraphicsColorPickingTex()),
								   static_cast<GLint>(cameraSystem->getNativeGraphicsDepthTex()),
								   static_cast<GLint>(cameraSystem->getNativeGraphicsNormalTex()),
								   static_cast<GLint>(cameraSystem->getNativeGraphicsPositionTex()),
								   static_cast<GLint>(cameraSystem->getNativeGraphicsAlbedoSpecTex()),
								   static_cast<GLint>(cameraSystem->getNativeGraphicsSSAOColorTex()),
								   static_cast<GLint>(cameraSystem->getNativeGraphicsSSAONoiseTex())};

		// select draw texture dropdown
		if (ImGui::BeginCombo("##DrawTexture", textureNames[activeTextureIndex]))
		{
			for (int n = 0; n < count; n++)
			{
				if (textures[n] == -1) {
					ImGui::PushItemFlag(ImGuiItemFlags_Disabled, true);
					ImGui::PushStyleVar(ImGuiStyleVar_Alpha, ImGui::GetStyle().Alpha * 0.5f);
				}

				bool is_selected = (textureNames[activeTextureIndex] == textureNames[n]);
				if (ImGui::Selectable(textureNames[n], is_selected)) {
					activeTextureIndex = n;
					
					if (is_selected) {
						ImGui::SetItemDefaultFocus();
					}
				}

				if (textures[n] == -1) {
					ImGui::PopItemFlag();
					ImGui::PopStyleVar();
				}
			}
			ImGui::EndCombo();
		}
		ImGui::SameLine();

		// whether to render gizmos or not
		if (ImGui::Checkbox("Gizmos", &gizmosChecked)) {

		}
		ImGui::SameLine();

		// editor rendering performance overlay
		if (ImGui::Checkbox("Perf", &overlayChecked)) {

		}
		ImGui::SameLine();

		// select transform gizmo movement mode
		if (ImGui::StampButton("T", translationModeActive))
		{
			translationModeActive = true;
			rotationModeActive = false;
			scaleModeActive = false;

			transformGizmo.setGizmoMode(GizmoMode::Translation);
		}
		ImGui::SameLine();

		if (ImGui::StampButton("R", rotationModeActive))
		{
			translationModeActive = false;
			rotationModeActive = true;
			scaleModeActive = false;

			transformGizmo.setGizmoMode(GizmoMode::Rotation);
		}
		ImGui::SameLine();

		if (ImGui::StampButton("S", scaleModeActive))
		{
			translationModeActive = false;
			rotationModeActive = false;
			scaleModeActive = true;

			transformGizmo.setGizmoMode(GizmoMode::Scale);
		}
		ImGui::SameLine();

		// editor camera settings
		if (ImGui::Button("Camera Settings"))
		{
			cameraSettingsClicked = true;
		}

		if (cameraSettingsClicked) {
			drawCameraSettingsPopup(cameraSystem, &cameraSettingsClicked);
		}

		// performance overlay
		if (overlayChecked) {
			drawPerformanceOverlay(cameraSystem);
		}

		// Update selected entity
		if (cameraSystem->isLeftMouseClicked() && !transformGizmo.isGizmoHighlighted()) {
			float nx = cameraSystem->getMousePosX() / (float)(sceneContentMax.x - sceneContentMin.x);
			float ny = cameraSystem->getMousePosY() / (float)(sceneContentMax.y - sceneContentMin.y);
			Guid transformId = cameraSystem->getTransformUnderMouse(nx, ny);

			Transform* transform = world->getComponentById<Transform>(transformId);

			if (transform != NULL) {
				clipboard.setSelectedItem(InteractionType::Entity, transform->getEntityId());
			}
			else {
				clipboard.setSelectedItem(InteractionType::None, Guid::INVALID);
			}
		}
			
		// draw transform gizmo if entity is selected
		if (clipboard.getSelectedType() == InteractionType::Entity) {
			Transform* transform = world->getComponent<Transform>(clipboard.getSelectedId());

			float width = (float)(sceneContentMax.x - sceneContentMin.x);
			float height = (float)(sceneContentMax.y - sceneContentMin.y);

			transformGizmo.update(cameraSystem, transform, width, height);
		}

		// Finally draw scene
		ImVec2 size = sceneContentMax;
		size.x -= sceneContentMin.x;
		size.y -= sceneContentMin.y;
		ImGui::Image((void*)(intptr_t)textures[activeTextureIndex], size, ImVec2(0, 1), ImVec2(1, 0));
	}
	ImGui::End();
}

void SceneView::drawPerformanceOverlay(PhysicsEngine::EditorCameraSystem* cameraSystem)
{
	static bool overlayOpened = false;
	static ImGuiWindowFlags overlayFlags = ImGuiWindowFlags_Tooltip |
		ImGuiWindowFlags_ChildWindow |
		ImGuiWindowFlags_NoTitleBar |
		ImGuiWindowFlags_AlwaysAutoResize |
		ImGuiWindowFlags_NoSavedSettings |
		ImGuiWindowFlags_NoResize |
		ImGuiWindowFlags_NoDocking |
		ImGuiWindowFlags_NoNav |
		ImGuiWindowFlags_NoMove;

	ImVec2 overlayPos = ImVec2(sceneContentMax.x, sceneContentMin.y);

	ImGui::SetNextWindowPos(overlayPos, ImGuiCond_Always, ImVec2(1.0f, 0.0f));
	ImGui::SetNextWindowBgAlpha(0.35f); // Transparent background
	if (ImGui::Begin("Editor Performance Overlay", &overlayOpened, overlayFlags))
	{
		ImGui::Text("Tris: %d\n", cameraSystem->getQuery().mTris);
		ImGui::Text("Verts: %d\n", cameraSystem->getQuery().mVerts);
		ImGui::Text("Draw calls: %d\n", cameraSystem->getQuery().mNumDrawCalls);
		ImGui::Text("Elapsed time: %f", cameraSystem->getQuery().mTotalElapsedTime);
		ImGui::Text("Window position: %f %f\n", windowPos.x, windowPos.y);
		//ImGui::Text("Content min: %f %f\n", contentMin.x, contentMin.y);
		//ImGui::Text("Content max: %f %f\n", contentMax.x, contentMax.y);
		ImGui::Text("Scene content min: %f %f\n", sceneContentMin.x, sceneContentMin.y);
		ImGui::Text("Scene content max: %f %f\n", sceneContentMax.x, sceneContentMax.y);
		ImGui::Text("Mouse Position: %d %d\n", cameraSystem->getMousePosX(), cameraSystem->getMousePosY());
		ImGui::Text("Normalized Mouse Position: %f %f\n", cameraSystem->getMousePosX() / (float)(sceneContentMax.x - sceneContentMin.x), cameraSystem->getMousePosY() / (float)(sceneContentMax.y - sceneContentMin.y));


		float width = (float)(sceneContentMax.x - sceneContentMin.x);
		float height = (float)(sceneContentMax.y - sceneContentMin.y);
		ImGui::Text("NDC: %f %f\n", 2 * (cameraSystem->getMousePosX() - 0.5f * width) / width, 2 * (cameraSystem->getMousePosY() - 0.5f * height) / height);

		ImGui::GetForegroundDrawList()->AddRect(sceneContentMin, sceneContentMax, 0xFFFF0000);

		perfQueue.addSample(cameraSystem->getQuery().mTotalElapsedTime);

		std::vector<float> perfData = perfQueue.getData();
		ImGui::PlotHistogram("##PerfPlot", &perfData[0], (int)perfData.size());
		//ImGui::PlotLines("Curve", &perfData[0], perfData.size());
	}
	ImGui::End();
}

void SceneView::drawCameraSettingsPopup(PhysicsEngine::EditorCameraSystem* cameraSystem, bool* cameraSettingsActive)
{
	static bool cameraSettingsWindowOpen = false;

	ImGui::SetNextWindowSize(ImVec2(430, 450), ImGuiCond_FirstUseEver);
	if (ImGui::Begin("Editor Camera Settings", cameraSettingsActive))
	{
		Viewport viewport = cameraSystem->getViewport();
		Frustum frustum = cameraSystem->getFrustum();

		// Viewport settings
		if (ImGui::InputInt("X", &viewport.mX)) {
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
		}

		// Frustum settings
		if (ImGui::InputFloat("FOV", &frustum.mFov)) {
			cameraSystem->setFrustum(frustum);
		}
		if (ImGui::InputFloat("Aspect Ratio", &frustum.mAspectRatio)) {
			cameraSystem->setFrustum(frustum);
		}
		if (ImGui::InputFloat("Near Plane", &frustum.mNearPlane)) {
			cameraSystem->setFrustum(frustum);
		}
		if (ImGui::InputFloat("Far Plane", &frustum.mFarPlane)) {
			cameraSystem->setFrustum(frustum);
		}

		// SSAO and render path
		int renderPath = static_cast<int>(cameraSystem->getRenderPath());
		int ssao = static_cast<int>(cameraSystem->getSSAO());

		const char* renderPathNames[] = { "Forward", "Deferred" };
		const char* ssaoNames[] = { "On", "Off" };

		if (ImGui::Combo("Render Path", &renderPath, renderPathNames, 2)) {
			cameraSystem->setRenderPath(static_cast<RenderPath>(renderPath));
		}

		if (ImGui::Combo("SSAO", &ssao, ssaoNames, 2)) {
			cameraSystem->setSSAO(static_cast<CameraSSAO>(ssao));
		}
	}

	ImGui::End();
}

bool SceneView::isFocused() const
{
	return focused;
}

bool SceneView::isHovered() const
{
	return hovered;
}

ImVec2 SceneView::getSceneContentMin() const
{
	return sceneContentMin;
}

ImVec2 SceneView::getSceneContentMax() const
{
	return sceneContentMax;
}

ImVec2 SceneView::getWindowPos() const
{
	return windowPos;
}