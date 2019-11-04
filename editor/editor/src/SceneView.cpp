#include "../include/SceneView.h"

#include "core/Log.h"

#include "../include/imgui/imgui.h"
#include "../include/imgui/imgui_impl_win32.h"
#include "../include/imgui/imgui_impl_opengl3.h"
#include "../include/imgui/imgui_internal.h"

using namespace PhysicsEngine;
using namespace PhysicsEditor;

SceneView::SceneView()
{
	focused = false;

	perfQueue.setNumberOfSamples(100);
}

SceneView::~SceneView()
{

}

void SceneView::render(PhysicsEngine::World* world, const char* textureNames[], const GLint textures[], int count, PhysicsEngine::GraphicsQuery query, bool isOpenedThisFrame)
{
	focused = false;
	static bool sceneViewActive = true;

	if (isOpenedThisFrame) {
		sceneViewActive = true;
	}

	if (!sceneViewActive) {
		return;
	}

	static bool gizmosChecked = false;
	static bool overlayChecked = false;

	ImGui::Begin("Scene View", &sceneViewActive);
	{
		focused = ImGui::IsWindowFocused();

		// select draw texture 
		static GLuint currentTexture = (GLuint)textures[0];
		static const char* currentTextureName = textureNames[0];

		if (ImGui::BeginCombo("##DrawTexture", currentTextureName))
		{
			for (int n = 0; n < count; n++)
			{
				if (textures[n] == -1) {
					ImGui::PushItemFlag(ImGuiItemFlags_Disabled, true);
					ImGui::PushStyleVar(ImGuiStyleVar_Alpha, ImGui::GetStyle().Alpha * 0.5f);
				}

				bool is_selected = (currentTextureName == textureNames[n]); // You can store your selection however you want, outside or inside your objects
				if (ImGui::Selectable(textureNames[n], is_selected)) {
					currentTextureName = textureNames[n];
					currentTexture = (GLuint)textures[n];
					world->debugView = n - 1;
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

		// draw selected texture
		ImVec2 windowPos = ImGui::GetWindowPos();
		ImVec2 contentMin = ImGui::GetWindowContentRegionMin();
		ImVec2 contentMax = ImGui::GetWindowContentRegionMax();

		contentMin.x += windowPos.x;
		contentMin.y += windowPos.y;
		contentMax.x += windowPos.x;
		contentMax.y += windowPos.y;

		// ImGui::GetForegroundDrawList()->AddRect(contentMin, contentMax, 0xFFFF0000);

		ImVec2 size = contentMax;
		size.x -= contentMin.x;
		size.y -= contentMin.y + 25; // account for the fact that Image will draw below buttons

		ImGui::Image((void*)(intptr_t)currentTexture, size, ImVec2(1, 1), ImVec2(0, 0));

		if (overlayChecked) {
			static bool overlayOpened = false;

			ImVec2 overlayPos = ImVec2(contentMax.x, contentMin.y + 25);

			ImGui::SetNextWindowPos(overlayPos, ImGuiCond_Always, ImVec2(1.0f, 0.0f));
			ImGui::SetNextWindowBgAlpha(0.35f); // Transparent background
			if (ImGui::Begin("Editor Performance Overlay", &overlayOpened, ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoDocking | ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoFocusOnAppearing | ImGuiWindowFlags_NoNav))
			{
				ImGui::Text("Tris: %d\n", query.tris);
				ImGui::Text("Verts: %d\n", query.verts);
				ImGui::Text("Draw calls: %d\n", query.numDrawCalls);
				ImGui::Text("Elapsed time: %f", query.totalElapsedTime);

				perfQueue.addSample(query.totalElapsedTime);

				std::vector<float> perfData = perfQueue.getData();
				ImGui::PlotHistogram("##PerfPlot", &perfData[0], (int)perfData.size());
				//ImGui::PlotLines("Curve", &perfData[0], perfData.size());
			}
			ImGui::End();
		}
	}
	ImGui::End();
}

bool SceneView::isFocused() const
{
	return focused;
}