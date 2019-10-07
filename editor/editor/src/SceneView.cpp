#include "../include/SceneView.h"


#include "../include/imgui/imgui.h"
#include "../include/imgui/imgui_impl_win32.h"
#include "../include/imgui/imgui_impl_opengl3.h"
#include "../include/imgui/imgui_internal.h"

using namespace PhysicsEditor;

SceneView::SceneView()
{
	focused = false;
}

SceneView::~SceneView()
{

}

void SceneView::render(const char* textureNames[], const GLuint textures[], int count, PhysicsEngine::GraphicsQuery query, bool isOpenedThisFrame)
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
		static GLuint currentTexture = textures[0];
		static const char* currentTextureName = textureNames[0];

		if (ImGui::BeginCombo("##DrawTexture", currentTextureName))
		{
			for (int n = 0; n < count; n++)
			{
				//ImGui::PushItemFlag(ImGuiItemFlags_Disabled, true);
				//ImGui::PushStyleVar(ImGuiStyleVar_Alpha, ImGui::GetStyle().Alpha * 0.5f);

				bool is_selected = (currentTextureName == textureNames[n]); // You can store your selection however you want, outside or inside your objects
				if (ImGui::Selectable(textureNames[n], is_selected)) {
					currentTextureName = textureNames[n];
					currentTexture = textures[n];
					if (is_selected) {
						ImGui::SetItemDefaultFocus();
					}
				}

				//ImGui::PopItemFlag();
				//ImGui::PopStyleVar();
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
		ImVec2 windowOrigin = ImGui::GetWindowPos();
		ImVec2 contentSize = ImGui::GetWindowContentRegionMax();
		contentSize.y -= 60; // how do I get the header height of a window?

		ImGui::Image((void*)(intptr_t)currentTexture, contentSize, ImVec2(1,1), ImVec2(0,0));
	}
	ImGui::End();

	if (overlayChecked) {
		static bool overlayOpened = false;

		ImGuiIO& io = ImGui::GetIO();
		ImGuiViewport* viewport = ImGui::GetMainViewport();
		ImVec2 window_pos = ImVec2((viewport->Pos.x + viewport->Size.x - 10.0f), (viewport->Pos.y + 10.0f));
		ImVec2 window_pos_pivot = ImVec2(1.0f, 0.0f);
		ImGui::SetNextWindowPos(window_pos, ImGuiCond_Always, window_pos_pivot);
		ImGui::SetNextWindowViewport(viewport->ID);

		ImGui::SetNextWindowBgAlpha(0.35f); // Transparent background
		if (ImGui::Begin("Editor Performance Overlay", &overlayOpened, ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoDocking | ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoFocusOnAppearing | ImGuiWindowFlags_NoNav))
		{
			ImGui::Text("Tris: %d\n", query.tris);
			ImGui::Text("Verts: %d\n", query.verts);
			ImGui::Text("Draw calls: %d\n", query.numDrawCalls);
			ImGui::Text("Elapsed time: %f", query.totalElapsedTime);
			/*ImGui::Text("Simple overlay\n" "in the corner of the screen.\n" "(right-click to change position)");
			ImGui::Separator();
			if (ImGui::IsMousePosValid())
				ImGui::Text("Mouse Position: (%.1f,%.1f)", 240, 300);
			else
				ImGui::Text("Mouse Position: <invalid>");*/
		}
		ImGui::End();
	}
}

bool SceneView::isFocused() const
{
	return focused;
}