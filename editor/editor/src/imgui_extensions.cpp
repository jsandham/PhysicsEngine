#include "../include/imgui_extensions.h"

#include "../include/imgui/imgui.h"
#include "../include/imgui/imgui_impl_win32.h"
#include "../include/imgui/imgui_impl_opengl3.h"
#include "../include/imgui/imgui_internal.h"

using namespace ImGui;

void ImGui::EnableDocking() // really this is rendering the main background docking window. Prob just make this a non imgui extension ToolBarWindow class? MainWindow class?
{
	static bool p_open = true;
	static bool opt_fullscreen_persistant = true;
	bool opt_fullscreen = opt_fullscreen_persistant;
	static ImGuiDockNodeFlags dockspace_flags = ImGuiDockNodeFlags_None;

	// We are using the ImGuiWindowFlags_NoDocking flag to make the parent window not dockable into,
	// because it would be confusing to have two docking targets within each others.
	ImGuiWindowFlags window_flags = ImGuiWindowFlags_MenuBar | ImGuiWindowFlags_NoDocking;
	if (opt_fullscreen)
	{
		ImGuiViewport* viewport = ImGui::GetMainViewport();
		ImGui::SetNextWindowPos(viewport->Pos);
		ImGui::SetNextWindowSize(viewport->Size);
		ImGui::SetNextWindowViewport(viewport->ID);

		ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
		ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
		window_flags |= ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove;
		window_flags |= ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoNavFocus;
	}

	// When using ImGuiDockNodeFlags_PassthruCentralNode, DockSpace() will render our background and handle the pass-thru hole, so we ask Begin() to not render a background.
	if (dockspace_flags & ImGuiDockNodeFlags_PassthruCentralNode)
		window_flags |= ImGuiWindowFlags_NoBackground;

	// Important: note that we proceed even if Begin() returns false (aka window is collapsed).
	// This is because we want to keep our DockSpace() active. If a DockSpace() is inactive,
	// all active windows docked into it will lose their parent and become undocked.
	// We cannot preserve the docking relationship between an active window and an inactive docking, otherwise
	// any change of dockspace/settings would lead to windows being stuck in limbo and never being visible.
	ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
	ImGui::Begin("DockSpace Demo", &p_open, window_flags);

	ImGui::Text("This is a non-docking/non-moving toolbar that is part of the main background window");
	ImGui::Separator();
	ImGui::Columns(2, "test", true);
	ImGui::Button("A");
	ImGui::SameLine();
	ImGui::Button("B");
	ImGui::NextColumn();
	ImGui::Button("C");
	ImGui::Columns(1, "test2", true);

	ImGui::PopStyleVar();

	if (opt_fullscreen)
		ImGui::PopStyleVar(2);

	// DockSpace
	ImGuiIO& io = ImGui::GetIO();
	if (io.ConfigFlags & ImGuiConfigFlags_DockingEnable)
	{
		ImGuiID dockspace_id = ImGui::GetID("MyDockSpace");
		ImGui::DockSpace(dockspace_id, ImVec2(0.0f, 0.0f), dockspace_flags);
	}

	ImGui::End();




	//ImGuiDockNodeFlags dockspaceFlags = ImGuiDockNodeFlags_None;
	//ImGuiID dockspaceID = ImGui::GetID(ID().c_str());
	//if (!ImGui::DockBuilderGetNode(dockspaceID)) {
	//	ImGui::DockBuilderRemoveNode(dockspaceID);
	//	ImGui::DockBuilderAddNode(dockspaceID, ImGuiDockNodeFlags_None);

	//	ImGuiID dock_main_id = dockspaceID;
	//	ImGuiID dock_up_id = ImGui::DockBuilderSplitNode(dock_main_id, ImGuiDir_Up, 0.05f, nullptr, &dock_main_id);
	//	ImGuiID dock_right_id = ImGui::DockBuilderSplitNode(dock_main_id, ImGuiDir_Right, 0.2f, nullptr, &dock_main_id);
	//	ImGuiID dock_left_id = ImGui::DockBuilderSplitNode(dock_main_id, ImGuiDir_Left, 0.2f, nullptr, &dock_main_id);
	//	ImGuiID dock_down_id = ImGui::DockBuilderSplitNode(dock_main_id, ImGuiDir_Down, 0.2f, nullptr, &dock_main_id);
	//	ImGuiID dock_down_right_id = ImGui::DockBuilderSplitNode(dock_down_id, ImGuiDir_Right, 0.6f, nullptr, &dock_down_id);

	//	ImGui::DockBuilderDockWindow("Actions", dock_up_id);
	//	ImGui::DockBuilderDockWindow("Hierarchy", dock_right_id);
	//	ImGui::DockBuilderDockWindow("Inspector", dock_left_id);
	//	ImGui::DockBuilderDockWindow("Console", dock_down_id);
	//	ImGui::DockBuilderDockWindow("Project", dock_down_right_id);
	//	ImGui::DockBuilderDockWindow("Scene", dock_main_id);

	//	// Disable tab bar for custom toolbar
	//	ImGuiDockNode* node = ImGui::DockBuilderGetNode(dock_up_id);
	//	node->LocalFlags |= ImGuiDockNodeFlags_NoTabBar;

	//	ImGui::DockBuilderFinish(dock_main_id);
	//}
	//ImGui::DockSpace(dockspaceID, ImVec2(0.0f, 0.0f), dockspaceFlags);
}

bool ImGui::BeginDropdown(std::string name, std::vector<std::string> values, int* selection)
{
	ImGui::SameLine(0.f, 0.f);
	ImGui::PushID(("##" + name).c_str());

	bool pressed = ImGui::Button(&name[0]);
	ImGui::PopID();

	if (pressed)
	{
		ImGui::OpenPopup(("##" + name).c_str());
	}

	if (ImGui::BeginPopup(("##" + name).c_str()))
	{
		std::vector<const char*> temp(values.size());
		for (size_t i = 0; i < values.size(); i++) {
			temp[i] = values[i].c_str();
		}
		if (ImGui::ListBox(("##" + name).c_str(), selection, &temp[0], (int)temp.size(), 4)) {
			ImGui::CloseCurrentPopup();
		}
		return true;
	}

	return false;
}

void ImGui::EndDropdown()
{
	ImGui::EndPopup();
}