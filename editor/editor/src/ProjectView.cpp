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

}

ProjectView::~ProjectView()
{

}

void ProjectView::render(std::string currentProjectPath, bool isOpenedThisFrame)
{
	static bool projectViewActive = true;

	if (isOpenedThisFrame) {
		projectViewActive = isOpenedThisFrame;
	}

	if (!projectViewActive) {
		return;
	}

	if (ImGui::Begin("Project View", &projectViewActive)) {

		if (currentProjectPath != "") {
			ImGui::Columns(2, "tree", true);


			ProjectNode* firstNode = new ProjectNode();
			firstNode->directoryName = "data";
			firstNode->directoryPath = currentProjectPath + "\\data";
			firstNode->isExpanded = false;

			drawProjectNodeRecursive(firstNode);

			// delete nodes
			std::stack<ProjectNode*> stack;
			stack.push(firstNode);
			while (!stack.empty()) {
				ProjectNode* current = stack.top();
				stack.pop();

				for (size_t i = 0; i < current->children.size(); i++) {
					stack.push(current->children[i]);
				}

				delete current;
			}

			ImGui::NextColumn();
			ImGui::Text("Hello");
			ImGui::Text("World");
			ImGui::Text("Here");
			ImGui::Text("I");
			ImGui::Text("Am");


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

	if (ImGui::TreeNodeEx(node->directoryName.c_str())) {
		node->isExpanded = true;

		// recurse for each sub directory
		for (size_t i = 0; i < subDirectoryPaths.size(); i++) {
			ProjectNode* child = new ProjectNode();
			child->directoryName = subDirectoryNames[i];
			child->directoryPath = subDirectoryPaths[i];
			child->isExpanded = false;

			node->children.push_back(child);

			drawProjectNodeRecursive(child);
		}

		ImGui::TreePop();
	}
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