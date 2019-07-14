#include "../include/FileBrowser.h"

#include "../include/imgui/imgui.h"
#include "../include/imgui/imgui_impl_win32.h"
#include "../include/imgui/imgui_impl_opengl3.h"
#include "../include/imgui/imgui_internal.h"

using namespace PhysicsEditor;

Filebrowser::Filebrowser()
{
	isVisible = false;
	wasVisible = false;

	currentPath = currentWorkingDirectory();
}

Filebrowser::~Filebrowser()
{

}

void Filebrowser::render()
{
	if (wasVisible != isVisible){
		wasVisible = isVisible;
		if (isVisible){
			currentFiles = getFilesInDirectory(currentPath);

			ImGui::OpenPopup("Filebrowser");
			ImGui::SetNextWindowSize(ImVec2(1000, 600));
		}
	}

	if (ImGui::BeginPopup("Filebrowser"))
	{
		std::vector<const char*> temp(currentFiles.size());
		for (size_t i = 0; i < currentFiles.size(); ++i)
		{
			temp[i] = currentFiles[i].data();
		} 
		//const char** listbox_items = temp.data();


		int selection = 0;
		const char* listbox_items[] = { "Apple", "Banana", "Cherry", "Kiwi", "Mango", "Orange", "Pineapple", "Strawberry", "Watermelon" };
		//ImGui::ListBox("listbox\n(single select)", &listbox_item_current, listbox_items, IM_ARRAYSIZE(listbox_items), 4);
		/*if (ImGui::ListBox("##", &selection, vector_file_items_getter, &currentFiles, currentFiles.size(), 10)) {*/
		if (ImGui::ListBox("##", &selection, listbox_items, IM_ARRAYSIZE(listbox_items), 4)) {
			//Update current path to the selected list item.
			//currentPath = currentFiles[selection];// .path;
			//m_currentPathIsDir = fs::is_directory(m_currentPath);

			//If the selection is a directory, repopulate the list with the contents of that directory.
			/*if (m_currentPathIsDir) {
				get_files_in_path(m_currentPath, m_filesInScope);
			}*/

		}

		bool reclaim_focus = false;
		char InputBuf[256];
		if (ImGui::InputText("File", InputBuf, IM_ARRAYSIZE(InputBuf), ImGuiInputTextFlags_EnterReturnsTrue | ImGuiInputTextFlags_CallbackCompletion | ImGuiInputTextFlags_CallbackHistory)){

		}

		ImGui::SameLine();
		if (ImGui::Button("Open")){

		}

		ImGui::SameLine();
		if (ImGui::Button("Cancel")){
			isVisible = false;
		}

		//if (ImGui::InputText("Input", InputBuf, IM_ARRAYSIZE(InputBuf), ImGuiInputTextFlags_EnterReturnsTrue | ImGuiInputTextFlags_CallbackCompletion | ImGuiInputTextFlags_CallbackHistory, &TextEditCallbackStub, (void*)this))
		//{
		//	/*char* s = InputBuf;
		//	Strtrim(s);
		//	if (s[0])
		//		ExecCommand(s);
		//	strcpy(s, "");
		//	reclaim_focus = true;*/
		//}

		/*ImGui::TextWrapped("Enter 'HELP' for help, press TAB to use text completion.");*/
		/*ImGui::TextWrapped(currentFiles[0].c_str());*/
		ImGui::TextWrapped(std::to_string(currentFiles.size()).c_str());
		ImGui::EndPopup();
	}
}