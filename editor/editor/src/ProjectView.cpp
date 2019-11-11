#include "../include/ProjectView.h"

#include "../include/imgui/imgui.h"
#include "../include/imgui/imgui_impl_win32.h"
#include "../include/imgui/imgui_impl_opengl3.h"
#include "../include/imgui/imgui_internal.h"

#include "../include/FileSystemUtil.h"

#include <stack>

#include "core/Log.h"

using namespace PhysicsEditor;

ProjectView::ProjectView()
{
	root = NULL;
	selected = NULL;
	projectViewActive = true;
}

ProjectView::~ProjectView()
{
	deleteProjectTree();
}

void ProjectView::render(std::string currentProjectPath, bool editorBecameActiveThisFrame, bool isOpenedThisFrame)
{
	if (isOpenedThisFrame) {
		projectViewActive = isOpenedThisFrame;
	}

	if (!projectViewActive) {
		return;
	}

	if (currentProjectPath != "") {
		if (root != NULL && root->directoryPath != currentProjectPath + "\\data" || root == NULL || editorBecameActiveThisFrame) {
			buildProjectTree(currentProjectPath);
		}
	}

	if (ImGui::Begin("Project View", &projectViewActive)) {


		/*ImVec2 size = ImVec2(5, 5);
		ImVec2 cursorPos = ImGui::GetMousePos();
		size.x += cursorPos.x;
		size.y += cursorPos.y;
		ImGui::GetForegroundDrawList()->AddRect(cursorPos, size, 0xFFFF0000);*/


		if (currentProjectPath != "") {
			ImGui::Columns(2, "ProjectViews", true);

			drawProjectTree();

			ImGui::NextColumn();
			ImGui::BeginChild("AAA");
			if (selected != NULL) {
				//ImGui::Text(selectedNodeDirectoryPath.c_str());
				std::vector<std::string> fileNames = PhysicsEditor::getFilesInDirectory(selected->directoryPath, false);
				for (int i = 0; i < fileNames.size(); i++) {
					ImGui::Text(fileNames[i].c_str());
					ImGui::Selectable(fileNames[i].c_str());
				}
			}
			ImGui::EndChild();

			ImGui::Columns(1);
			ImGui::Separator();
		}
	}

	ImGui::End();
}

void ProjectView::deleteProjectTree()
{
	if (root == NULL) { return; }

	selected = NULL;

	nodes.clear();

	std::stack<ProjectNode*> stack;
	stack.push(root);
	while (!stack.empty()) {
		ProjectNode* current = stack.top();
		stack.pop();

		for (size_t i = 0; i < current->children.size(); i++) {
			stack.push(current->children[i]);
		}

		delete current;
	}
}

void ProjectView::buildProjectTree(std::string currentProjectPath)
{
	deleteProjectTree();

	int id = -1;

	root = new ProjectNode();
	root->id = ++id;
	root->parent = NULL;
	root->directoryName = "data";
	root->directoryPath = currentProjectPath +"\\data";
	root->isExpanded = false;

	nodes.push_back(root);

	std::stack<ProjectNode*> stack;
	stack.push(root);

	while (!stack.empty()) {
		ProjectNode* current = stack.top();
		stack.pop();

		// find directories that exist in the current directory
		std::vector<std::string> subDirectoryPaths = PhysicsEditor::getDirectoriesInDirectory(current->directoryPath, true);
		std::vector<std::string> subDirectoryNames = PhysicsEditor::getDirectoriesInDirectory(current->directoryPath, false);

		current->isExpanded = true;

		// recurse for each sub directory
		for (size_t i = 0; i < subDirectoryPaths.size(); i++) {
			ProjectNode* child = new ProjectNode();
			child->id = ++id;
			child->parent = current;
			child->directoryName = subDirectoryNames[i];
			child->directoryPath = subDirectoryPaths[i];
			child->isExpanded = false;

			current->children.push_back(child);

			stack.push(child);
			nodes.push_back(child);
		}
	}
}

void ProjectView::drawProjectTree()
{
	if (root != NULL) {
		drawProjectNodeRecursive(root);
	}
}

void ProjectView::drawProjectNodeRecursive(ProjectNode* node)
{
	ImGuiTreeNodeFlags node_flags = ImGuiTreeNodeFlags_None;
	if (node->children.size() == 0) {
		node_flags |= ImGuiTreeNodeFlags_Leaf;
	}

	if (ImGui::IsItemHovered()) {
		node_flags |= ImGuiTreeNodeFlags_Selected;
	}

	bool open = ImGui::TreeNodeEx(node->directoryName.c_str(), node_flags);

	//if (ImGui::IsItemHovered()) {
	//	ImVec2 windowPos = ImGui::GetWindowPos();
	//	ImVec2 contentsize = ImGui::GetItemRectSize();
	//	ImVec2 contentMin = ImGui::GetItemRectMin();
	//	ImVec2 contentMax = ImGui::GetItemRectMax();
	//	//ImVec2 contentMin = ImGui::GetWindowContentRegionMin();
	//	//ImVec2 contentMax = ImGui::GetWindowContentRegionMax();

	//	//ImVec2 cursorPos = ImGui::GetCursorPos();

	//	/*contentMin.x += cursorPos.x;
	//	contentMin.y += cursorPos.y;
	//	contentMax.x += cursorPos.x;
	//	contentMax.y += cursorPos.y;*/

	//	//contentsize.x += cursorPos.x;
	//	//contentsize.y += cursorPos.y;

	//	ImGui::GetForegroundDrawList()->AddRect(contentMin, contentMax, 0xFFFF0000);
	//	//ImGui::GetForegroundDrawList()->AddRect(cursorPos, contentsize, 0xFFFF0000);
	//}

	if (ImGui::IsItemClicked()) {
		selected = node;
		//selectedNodeDirectoryPath = node->directoryPath;
		//ImGui::Text(node->directoryName.c_str());
		// Some processing...
	}

	if (ImGui::BeginDragDropSource()) {
		ImGui::SetDragDropPayload("DND_DEMO_CELL", &node, sizeof(ProjectNode*));
		//ImGui::SetDragDropPayload("DND_DEMO_CELL", &node->id, sizeof(int));
		std::string message = node->directoryName + " " + std::to_string(node->id) + "\n";
		PhysicsEngine::Log::info(message.c_str());
		ImGui::EndDragDropSource();
	}
	
	if (ImGui::BeginDragDropTarget()) {
		if (const ImGuiPayload* payload = ImGui::AcceptDragDropPayload("DND_DEMO_CELL")) {
			/*IM_ASSERT(payload->DataSize == sizeof(int));
			int payloadId = *(int*)payload->Data;
			std::string message = node->directoryName + " " + std::to_string(node->id) + "\n";
			PhysicsEngine::Log::info(message.c_str());*/
			
			IM_ASSERT(payload->DataSize == sizeof(ProjectNode*));
			ProjectNode* payloadNode = *(ProjectNode**)payload->Data;
			std::string message1 = node->directoryName + " " + std::to_string(node->id) + "\n";
			std::string message2 = payloadNode->directoryName + " " + std::to_string(payloadNode->id) + "\n";
			PhysicsEngine::Log::info(message1.c_str());
			PhysicsEngine::Log::info(message2.c_str());

			// detach payload node from its parent
			ProjectNode* payloadParent = payloadNode->parent;
			if (payloadParent != NULL) {
				for (size_t i = payloadParent->children.size() - 1; i >= 0; i--) {
					if (payloadParent->children[i] == payloadNode) {
						payloadParent->children.erase(payloadParent->children.begin() + i);
						payloadNode->parent = NULL;
						break;
					}
				}
			}

			// attach payload to new parent
			node->children.push_back(payloadNode);
			payloadNode->parent = node;
		}
		ImGui::EndDragDropTarget();
	}

	if (open) {
		node->isExpanded = true;

		// recurse for each sub directory
		for (size_t i = 0; i < node->children.size(); i++) {
			node->children[i]->isExpanded = false;

			drawProjectNodeRecursive(node->children[i]);
		}

		ImGui::TreePop();
	}


	//if (ImGui::TreeNodeEx(node->directoryName.c_str(), node_flags)) {
	//	node->isExpanded = true;

	//	// recurse for each sub directory
	//	for (size_t i = 0; i < node->children.size(); i++) {
	//		node->children[i]->isExpanded = false;

	//		drawProjectNodeRecursive(node->children[i]);
	//	}

	//	ImGui::TreePop();
	//}
}

//
//if (ImGui::IsItemClicked()) {
//	selectedNodeDirectoryPath = node->directoryPath;
//	//ImGui::Text(node->directoryName.c_str());
//	// Some processing...
//}
//
//if (ImGui::BeginDragDropSource()) {
//	ImGui::SetDragDropPayload("DND_DEMO_CELL", node, sizeof(ProjectNode*));
//	ImGui::Text(node->directoryName.c_str());
//	ImGui::EndDragDropSource();
//}
//
//if (ImGui::BeginDragDropTarget()) {
//	if (const ImGuiPayload* payload = ImGui::AcceptDragDropPayload("DND_DEMO_CELL")) {
//		IM_ASSERT(payload->DataSize == sizeof(ProjectNode*));
//		ProjectNode* payloadNode = (ProjectNode*)payload->Data;
//		ImGui::Text(node->directoryName.c_str());
//		ImGui::Text(payloadNode->directoryName.c_str());
//		//node->children.push_back(payloadNode);
//		//const char* tmp = names[n];
//		//names[n] = names[payload_n];
//		//names[payload_n] = tmp;
//	}
//	ImGui::EndDragDropTarget();
//}


//ImGui::Begin("Splitter test");
//
//static float w = 200.0f;
//static float h = 300.0f;
//ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(0, 0));
//ImGui::BeginChild("child1", ImVec2(w, h), true);
//ImGui::EndChild();
//ImGui::SameLine();
//ImGui::InvisibleButton("vsplitter", ImVec2(8.0f, h));
//if (ImGui::IsItemActive())
//w += ImGui::GetIO().MouseDelta.x;
//ImGui::SameLine();
//ImGui::BeginChild("child2", ImVec2(0, h), true);
//ImGui::EndChild();
//ImGui::InvisibleButton("hsplitter", ImVec2(-1, 8.0f));
//if (ImGui::IsItemActive())
//h += ImGui::GetIO().MouseDelta.y;
//ImGui::BeginChild("child3", ImVec2(0, 0), true);
//ImGui::EndChild();
//ImGui::PopStyleVar();
//
//ImGui::End();




















//// find directories that exist in the current directory
//std::vector<std::string> subDirectoryPaths = PhysicsEditor::getDirectoriesInDirectory(node->directoryPath, true);
//std::vector<std::string> subDirectoryNames = PhysicsEditor::getDirectoriesInDirectory(node->directoryPath, false);
//
//ImGuiTreeNodeFlags node_flags = ImGuiTreeNodeFlags_None;
//if (subDirectoryNames.size() == 0) {
//	node_flags |= ImGuiTreeNodeFlags_Leaf;
//}
//
//bool treeNodeOpen = ImGui::TreeNodeEx(node->directoryName.c_str(), node_flags);
//
//if (ImGui::IsItemClicked()) {
//	selectedNodeDirectoryPath = node->directoryPath;
//	//ImGui::Text(node->directoryName.c_str());
//	// Some processing...
//}
//
//if (ImGui::BeginDragDropSource()) {
//	ImGui::SetDragDropPayload("DND_DEMO_CELL", node, sizeof(ProjectNode*));
//	ImGui::Text(node->directoryName.c_str());
//	ImGui::EndDragDropSource();
//}
//
//if (ImGui::BeginDragDropTarget()) {
//	if (const ImGuiPayload* payload = ImGui::AcceptDragDropPayload("DND_DEMO_CELL")) {
//		IM_ASSERT(payload->DataSize == sizeof(ProjectNode*));
//		ProjectNode* payloadNode = (ProjectNode*)payload->Data;
//		ImGui::Text(node->directoryName.c_str());
//		ImGui::Text(payloadNode->directoryName.c_str());
//		//node->children.push_back(payloadNode);
//		//const char* tmp = names[n];
//		//names[n] = names[payload_n];
//		//names[payload_n] = tmp;
//	}
//	ImGui::EndDragDropTarget();
//}
//
//if (treeNodeOpen) {
//	node->isExpanded = true;
//
//	// recurse for each sub directory
//	for (size_t i = 0; i < subDirectoryPaths.size(); i++) {
//		ProjectNode* child = new ProjectNode();
//		child->parent = node;
//		child->directoryName = subDirectoryNames[i];
//		child->directoryPath = subDirectoryPaths[i];
//		child->isExpanded = false;
//
//		node->children.push_back(child);
//
//		drawProjectNodeRecursive(child);
//	}
//
//	ImGui::TreePop();
//}