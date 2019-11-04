#include "../include/ProjectView.h"

#include "../include/imgui/imgui.h"
#include "../include/imgui/imgui_impl_win32.h"
#include "../include/imgui/imgui_impl_opengl3.h"
#include "../include/imgui/imgui_internal.h"

#include "../include/FileSystemUtil.h"

#include <stack>

using namespace PhysicsEditor;

ProjectView::ProjectView()
{
	selectedNodeDirectoryPath = "";
}

ProjectView::~ProjectView()
{

}

void ProjectView::render(std::string currentProjectPath, bool editorApplicationActive, bool isOpenedThisFrame)
{
	static bool projectViewActive = true;

	if (isOpenedThisFrame) {
		projectViewActive = isOpenedThisFrame;
	}

	if (!projectViewActive) {
		return;
	}

	if (!prevEditorApplicationActive && editorApplicationActive) {
		rebuildProjectTree();
	}

	prevEditorApplicationActive = editorApplicationActive;

	if (ImGui::Begin("Project View", &projectViewActive)) {

		if (currentProjectPath != "") {
			ImGui::Columns(2, "ProjectViews", true);

			ProjectNode* rootNode = new ProjectNode();
			rootNode->parent = NULL;
			rootNode->directoryName = "data";
			rootNode->directoryPath = currentProjectPath + "\\data";
			rootNode->isExpanded = false;

			drawProjectNodeRecursive(rootNode);

			ImGui::NextColumn();
			if (selectedNodeDirectoryPath.length() > 0) {
				//ImGui::Text(selectedNodeDirectoryPath.c_str());
				std::vector<std::string> fileNames = PhysicsEditor::getFilesInDirectory(selectedNodeDirectoryPath, false);
				for (int i = 0; i < fileNames.size(); i++) {
					ImGui::Text(fileNames[i].c_str());
				}
			}

			// delete nodes
			std::stack<ProjectNode*> stack;
			stack.push(rootNode);
			while (!stack.empty()) {
				ProjectNode* current = stack.top();
				stack.pop();

				for (size_t i = 0; i < current->children.size(); i++) {
					stack.push(current->children[i]);
				}

				delete current;
			}


			ImGui::Columns(1);
			ImGui::Separator();
		}
	}

	ImGui::End();
}

void ProjectView::drawProjectNodeRecursive(ProjectNode* node)
{
	// find directories that exist in the current directory
	std::vector<std::string> subDirectoryPaths = PhysicsEditor::getDirectoriesInDirectory(node->directoryPath, true);
	std::vector<std::string> subDirectoryNames = PhysicsEditor::getDirectoriesInDirectory(node->directoryPath, false);

	ImGuiTreeNodeFlags node_flags = ImGuiTreeNodeFlags_None;
	if (subDirectoryNames.size() == 0) {
		node_flags |= ImGuiTreeNodeFlags_Leaf;
	}

	bool treeNodeOpen = ImGui::TreeNodeEx(node->directoryName.c_str(), node_flags);

	if (ImGui::IsItemClicked()) {
		selectedNodeDirectoryPath = node->directoryPath;
		//ImGui::Text(node->directoryName.c_str());
		// Some processing...
	}

	if (ImGui::BeginDragDropSource()) {
		ImGui::SetDragDropPayload("DND_DEMO_CELL", node, sizeof(ProjectNode*));
		ImGui::Text(node->directoryName.c_str());
		ImGui::EndDragDropSource();
	}

	if (ImGui::BeginDragDropTarget()) {
		if (const ImGuiPayload* payload = ImGui::AcceptDragDropPayload("DND_DEMO_CELL")) {
			IM_ASSERT(payload->DataSize == sizeof(ProjectNode*));
			ProjectNode* payloadNode = (ProjectNode*)payload->Data;
			//node->children.push_back(payloadNode);
			//const char* tmp = names[n];
			//names[n] = names[payload_n];
			//names[payload_n] = tmp;
		}
		ImGui::EndDragDropTarget();
	}

	if (treeNodeOpen) {
		node->isExpanded = true;

		// recurse for each sub directory
		for (size_t i = 0; i < subDirectoryPaths.size(); i++) {
			ProjectNode* child = new ProjectNode();
			child->parent = node;
			child->directoryName = subDirectoryNames[i];
			child->directoryPath = subDirectoryPaths[i];
			child->isExpanded = false;

			node->children.push_back(child);

			drawProjectNodeRecursive(child);
		}

		ImGui::TreePop();
	}






	//if (ImGui::TreeNodeEx(node->directoryName.c_str(), node_flags)) {
	//	node->isExpanded = true;

	//	// recurse for each sub directory
	//	for (size_t i = 0; i < subDirectoryPaths.size(); i++) {
	//		ProjectNode* child = new ProjectNode();
	//		child->directoryName = subDirectoryNames[i];
	//		child->directoryPath = subDirectoryPaths[i];
	//		child->isExpanded = false;

	//		node->children.push_back(child);

	//		drawProjectNodeRecursive(child);
	//	}

	//	ImGui::TreePop();
	//}


}


void ProjectView::rebuildProjectTree()
{

}
















//std::stack<ProjectNode*> stack;

//ProjectNode* firstNode = new ProjectNode();
//firstNode->directoryName = "data";
//firstNode->directoryPath = currentProjectPath + "\\data";
//firstNode->isExpanded = false;

//stack.push(firstNode);

//while (!stack.empty()) {
//	ProjectNode* current = stack.top();
//	stack.pop();

//	if (ImGui::TreeNode(current->directoryName.c_str())) {
//		current->isExpanded = true;
//	}
//	//current->renderNode();

//	if (current->isExpanded) {
//		// find directories that exist in the current directory
//		std::vector<std::string> subDirectoryPaths = PhysicsEditor::getDirectoriesInDirectory(current->directoryPath, true);
//
//		// add a node for each onto the stack
//		for (size_t i = 0; i < subDirectoryPaths.size(); i++) {
//			ProjectNode* child = new ProjectNode();
//			child->directoryName = subDirectoryPaths[i];
//			child->directoryPath = subDirectoryPaths[i];
//			child->isExpanded = false;

//			current->children.push_back(child);

//			stack.push(child);
//		}

//	}
//}