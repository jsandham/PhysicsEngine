#include "../../include/views/ProjectView.h"
#include "../../include/ProjectDatabase.h"

#include <algorithm>
#include <stack>
#include <queue>
#include <assert.h>

#include "../../include/imgui/imgui_extensions.h"
#include "../../include/IconsFontAwesome4.h"

#include "core/Guid.h"
#include "core/Log.h"

using namespace PhysicsEditor;

ProjectNode::ProjectNode()
{
	mParent = nullptr;
	mDirectoryPath = std::filesystem::path();
	mDirectoryName = std::string();
	mDirectoryLabelEmpty = std::string();
	mDirectoryLabelNonEmpty = std::string();

	mFileLabels.reserve(200);
	mFileNames.reserve(200);
	mFileTypes.reserve(200);
}

ProjectNode::ProjectNode(const std::filesystem::path& path)
{
	mParent = nullptr;
	mDirectoryPath = path;
	mDirectoryName = path.filename().string();
	mDirectoryLabelEmpty = std::string(ICON_FA_FOLDER_O) + " " + mDirectoryName;
	mDirectoryLabelNonEmpty = std::string(ICON_FA_FOLDER) + " " + mDirectoryName;

	mFileLabels.reserve(200);
	mFileNames.reserve(200);
	mFileTypes.reserve(200);
}

std::string ProjectNode::getDirectoryLabel() const
{
	if (mChildren.size() == 0)
	{
		if (mFileNames.size() == 0)
		{
			return mDirectoryLabelEmpty;
		}
	}

	return mDirectoryLabelNonEmpty;
}

std::filesystem::path ProjectNode::getFilePath(size_t index) const
{
	assert(index < mFileNames.size());
	return mDirectoryPath / mFileNames[index];
}

ProjectNode* ProjectNode::addDirectory(const std::filesystem::path& path)
{
	ProjectNode* node = new ProjectNode();
	node->mParent = this;
	node->mDirectoryPath = path;
	node->mDirectoryName = path.filename().string();
	node->mDirectoryLabelEmpty = std::string(ICON_FA_FOLDER_O) + " " + node->mDirectoryName;
	node->mDirectoryLabelNonEmpty = std::string(ICON_FA_FOLDER) + " " + node->mDirectoryName;

	mChildren.push_back(node);

	return node;
}

void ProjectNode::addFile(const std::filesystem::path& path)
{
	std::string filename = path.filename().string();
	std::string extension = filename.substr(filename.find_last_of(".") + 1);

	std::string label;
	InteractionType type;
	if (extension.length() >= 1)
	{
		if (extension[0] == 's')
		{
			if (extension == "scene")
			{
				label = std::string(ICON_FA_MAXCDN);
				type = InteractionType::Scene;
			}
			else if (extension == "shader")
			{
				label = std::string(ICON_FA_AREA_CHART);
				type = InteractionType::Shader;
			}
		}
		else if (extension[0] == 'm')
		{
			if (extension == "material")
			{
				label = std::string(ICON_FA_MAXCDN);
				type = InteractionType::Material;
			}
			else if (extension == "mesh")
			{
				label = std::string(ICON_FA_CODEPEN);
				type = InteractionType::Mesh;
			}
		}
		else if (extension == "texture")
		{
			label = std::string(ICON_FA_FILE_IMAGE_O);
			type = InteractionType::Texture2D;
		}
		else if (extension == "rendertexture")
		{
			label = std::string(ICON_FA_AREA_CHART);
			type = InteractionType::RenderTexture;
		}
		else if (extension == "cubemap")
		{
			label = std::string(ICON_FA_AREA_CHART);
			type = InteractionType::Cubemap;
		}
		else
		{
			label = std::string(ICON_FA_FILE);
			type = InteractionType::File;
		}
	}
	else
	{
		label = std::string(ICON_FA_FILE);
		type = InteractionType::File;
	}

	mFileLabels.push_back(label + " " + filename);
	mFileNames.push_back(filename);
	mFileTypes.push_back(type);
}

ProjectTree::ProjectTree()
{
	mRoot = nullptr;
}

ProjectTree::~ProjectTree()
{
	deleteProjectTree();
}

void ProjectTree::buildProjectTree(const std::filesystem::path& projectPath)
{
	deleteProjectTree();

	mRoot = new ProjectNode(projectPath / "data");

	std::queue<ProjectNode*> queue;
	queue.push(mRoot);

	while (!queue.empty())
	{
		ProjectNode* current = queue.front();
		queue.pop();

		for (const auto& entry : std::filesystem::directory_iterator(current->mDirectoryPath))
		{
			if (entry.is_regular_file())
			{
				current->addFile(entry.path());
			}
			else if (entry.is_directory())
			{
				ProjectNode* child = current->addDirectory(entry.path());
				queue.push(child);
			}
		}
	}
}

void ProjectTree::deleteProjectTree()
{
	if (mRoot != nullptr)
	{
		std::queue<ProjectNode*> queue;
		queue.push(mRoot);

		while (!queue.empty())
		{
			ProjectNode* current = queue.front();
			queue.pop();

			for (size_t i = 0; i < current->mChildren.size(); i++)
			{
				queue.push(current->mChildren[i]);
			}

			delete current;
		}
	}
}

// move source to be a child of target
void ProjectTree::move(ProjectNode* target, ProjectNode* source)
{
	assert(source != nullptr);
	assert(target != nullptr);

	if (source->mParent != nullptr)
	{
		// Ensure that source is not a parent of target
		{
			std::stack<ProjectNode*> stack;
			stack.push(target);

			while (!stack.empty())
			{
				ProjectNode* current = stack.top();
				stack.pop();

				if (current == source)
				{
					return;
				}

				if (current->mParent != nullptr)
				{
					stack.push(current->mParent);
				}
			}
		}

		// Find the index of source parent children that source resides at
		int index = -1;
		for (size_t j = 0; j < source->mParent->mChildren.size(); j++)
		{
			if (source->mParent->mChildren[j] == source)
			{
				index = j;
				break;
			}
		}

		assert(index != -1);

		// Detach source from its parent
		source->mParent->mChildren[index] = source->mParent->mChildren[source->mParent->mChildren.size() - 1];
		source->mParent->mChildren.resize(source->mParent->mChildren.size() - 1);
		source->mParent = nullptr;

		// Attach source as a child of target
		target->mChildren.push_back(source);
		source->mParent = target;

		// Fix directory paths of all children of source
		{
			std::stack<ProjectNode*> stack;
			stack.push(source);

			while (!stack.empty())
			{
				ProjectNode* current = stack.top();
				stack.pop();

				current->mDirectoryPath = current->mParent->mDirectoryPath / current->mDirectoryPath.filename();

				for (size_t i = 0; i < current->mChildren.size(); i++)
				{
					stack.push(current->mChildren[i]);
				}
			}
		}
	}
}

void ProjectTree::move(ProjectNode* target, ProjectNode* source, const std::string& sourceFilename)
{
	assert(target != nullptr);
	assert(source != nullptr);

	// Remove source file from sources lists of files
	int index = -1;
	for (size_t i = 0; i < source->mFileNames.size(); i++)
	{
		if (source->mFileNames[i] == sourceFilename)
		{
			index = i;
			break;
		}
	}

	assert(index != -1);

	std::string sourceFileLabel = source->mFileLabels[index];
	InteractionType sourceFileType = source->mFileTypes[index];

	source->mFileNames[index] = source->mFileNames[source->mFileNames.size() - 1];
	source->mFileLabels[index] = source->mFileLabels[source->mFileLabels.size() - 1];
	source->mFileTypes[index] = source->mFileTypes[source->mFileTypes.size() - 1];

	source->mFileNames.resize(source->mFileNames.size() - 1);
	source->mFileLabels.resize(source->mFileLabels.size() - 1);
	source->mFileTypes.resize(source->mFileTypes.size() - 1);

	// Add source file as to targets list of files
	target->mFileNames.push_back(sourceFilename);
	target->mFileLabels.push_back(sourceFileLabel);
	target->mFileTypes.push_back(sourceFileType);
}

ProjectView::ProjectView() : mOpen(true), mBuildRequired(true)
{
	mHighlightedType = InteractionType::None;
	mHighlightedPath = std::filesystem::path();
	mHoveredPath = std::filesystem::path();

	mSelectedDirectoryPath = std::filesystem::path();
	mSelectedFilePath = std::filesystem::path();
}

ProjectView::~ProjectView()
{

}

void ProjectView::init(Clipboard& clipboard)
{
}

void ProjectView::update(Clipboard& clipboard, bool isOpenedThisFrame)
{
	if (isOpenedThisFrame)
	{
		mOpen = true;
	}

	if (!mOpen)
	{
		return;
	}

	if (ImGui::Begin("ProjectView", &mOpen))
	{
		if (ImGui::GetIO().MouseClicked[1] && ImGui::IsWindowHovered())
		{
			ImGui::SetWindowFocus("ProjectView");
		}
	}

	if (clipboard.mProjectOpened)
	{
		if (mBuildRequired)
		{
			mProjectTree.buildProjectTree(clipboard.getProjectPath());
			mBuildRequired = false;
		}

		mFilter.Draw("Filter", -100.0f);

		ImVec2 WindowSize = ImGui::GetWindowSize();

		static float ratio = 0.5f;

		float sz1 = ratio * WindowSize.x;
		float sz2 = (1.0f - ratio) * WindowSize.x;

		ImGui::Splitter(true, 8.0f, &sz1, &sz2, 8, 8, WindowSize.y);

		ratio = sz1 / WindowSize.x;

		ImGuiWindowFlags flags =
			ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoDocking;

		if (ImGui::BeginChild("LeftPane", ImVec2(sz1, WindowSize.y), true, flags))
		{
			drawLeftPane();
		}
		ImGui::EndChild();

		ImGui::SameLine();

		if (ImGui::BeginChild("RightPane", ImVec2(sz2, WindowSize.y), true, flags))
		{
			drawRightPane(clipboard);
		}
		ImGui::EndChild();
	}

	executeCommands();

	ImGui::End();
}

void ProjectView::executeCommands()
{
	while (!mCommandQueue.empty())
	{
		Command command = mCommandQueue.front();
		mCommandQueue.pop();

		std::stack<ProjectNode*> stack;
		stack.push(mProjectTree.mRoot);

		ProjectNode* target = nullptr;
		ProjectNode* source = nullptr;

		while (!stack.empty())
		{
			ProjectNode* current = stack.top();
			stack.pop();

			if (command.mDragDropType == DragDropType::Folder)
			{
				if (current->mDirectoryPath == command.mSource)
				{
					source = current;
				}
			}
			else
			{
				if (current->mDirectoryPath == command.mSource.parent_path())
				{
					source = current;
				}
			}

			if (current->mDirectoryPath == command.mTarget)
			{
				target = current;
			}

			if (source != nullptr && target != nullptr)
			{
				break;
			}

			for (size_t j = 0; j < current->mChildren.size(); j++)
			{
				stack.push(current->mChildren[j]);
			}
		}

		if (source != nullptr && target != nullptr)
		{
			if (command.mDragDropType == DragDropType::Folder)
			{
				mProjectTree.move(target, source);
			}
			else
			{
				mProjectTree.move(target, source, command.mSource.filename().string());
			}

			ProjectDatabase::rename(command.mSource, command.mTarget / command.mSource.filename());
		}
	}
}

constexpr auto DragDropTypesToString(DragDropType type)
{
	switch (type)
	{
	case DragDropType::Folder:
		return "FOLDER_PATH";
	case DragDropType::Material:
		return "MATERIAL_PATH";
	case DragDropType::Cubemap:
		return "CUBEMAP_PATH";
	case DragDropType::Scene:
		return "SCENE_PATH";
	}
}

constexpr auto InteractionTypeToDragAndDropTypeString(InteractionType type)
{
	switch (type)
	{
	case InteractionType::Cubemap:
		return "CUBEMAP_PATH";
	case InteractionType::Texture2D:
		return "TEXTURE2D_PATH";
	case InteractionType::Mesh:
		return "MESH_PATH";
	case InteractionType::Material:
		return "MATERIAL_PATH";
	case InteractionType::Scene:
		return "SCENE_PATH";
	case InteractionType::Shader:
		return "SHADER_PATH";
	case InteractionType::File:
		return "FILE_PATH";
	}
}

void ProjectView::drawLeftPane()
{
	drawProjectNodeRecursive(mProjectTree.mRoot);
}

void ProjectView::drawRightPane(Clipboard& clipboard)
{
	std::vector<ProjectNode*> directories;

	std::vector<std::string> fileLabels;
	std::vector<std::filesystem::path> filePaths;
	std::vector<InteractionType> fileTypes;

	// Determine directories and files to be drawn in right pane
	if (mFilter.IsActive())
	{
		std::stack<ProjectNode*> stack;

		stack.push(mProjectTree.mRoot);

		while (!stack.empty())
		{
			ProjectNode* current = stack.top();
			stack.pop();

			if (mFilter.PassFilter(current->mDirectoryName.c_str()))
			{
				directories.push_back(current);
			}

			for (size_t j = 0; j < current->mFileNames.size(); j++)
			{
				if (mFilter.PassFilter(current->mFileNames[j].c_str()))
				{
					fileLabels.push_back(current->mFileLabels[j]);
					filePaths.push_back(current->getFilePath(j));
					fileTypes.push_back(current->mFileTypes[j]);
				}
			}

			for (size_t j = 0; j < current->mChildren.size(); j++)
			{
				stack.push(current->mChildren[j]);
			}
		}
	}
	else
	{
		std::stack<ProjectNode*> stack;

		stack.push(mProjectTree.mRoot);

		while (!stack.empty())
		{
			ProjectNode* current = stack.top();
			stack.pop();

			if (mSelectedDirectoryPath.compare(current->mDirectoryPath) == 0)
			{
				directories = current->mChildren;
				fileLabels = current->mFileLabels;

				for (size_t j = 0; j < current->mFileNames.size(); j++)
				{
					filePaths.push_back(current->getFilePath(j));
				}
				fileTypes = current->mFileTypes;

				break;
			}

			for (size_t j = 0; j < current->mChildren.size(); j++)
			{
				stack.push(current->mChildren[j]);
			}
		}
	}

	// draw directories in right pane
	for (size_t i = 0; i < directories.size(); i++)
	{
		if (ImGui::Selectable(directories[i]->getDirectoryLabel().c_str(), mHighlightedPath.compare(directories[i]->mDirectoryPath) == 0, ImGuiSelectableFlags_AllowDoubleClick))
		{
			mHighlightedType = InteractionType::Folder;
			mHighlightedPath = directories[i]->mDirectoryPath;

			if (ImGui::IsMouseDoubleClicked(0))
			{
				mSelectedDirectoryPath = directories[i]->mDirectoryPath;

				mFilter.Clear();
			}
		}

		if (ImGui::IsItemHovered())
		{
			mHoveredPath = directories[i]->mDirectoryPath;
		}

		if (ImGui::BeginDragDropSource())
		{
			std::string directoryPath = directories[i]->mDirectoryPath.string();

			const void* data = static_cast<const void*>(directoryPath.c_str());

			ImGui::SetDragDropPayload(DragDropTypesToString(DragDropType::Folder), data, directoryPath.length() + 1);
			ImGui::Text(directoryPath.c_str());
			ImGui::EndDragDropSource();
		}

		if (ImGui::BeginDragDropTarget())
		{
			for (int j = 0; j < (int)DragDropType::Count; j++)
			{
				const ImGuiPayload* payload = ImGui::AcceptDragDropPayload(DragDropTypesToString(static_cast<DragDropType>(j)));
				if (payload != nullptr)
				{
					const char* data = static_cast<const char*>(payload->Data);

					Command command;
					command.mSource = std::string(data);
					command.mTarget = directories[i]->mDirectoryPath;
					command.mDragDropType = static_cast<DragDropType>(j);

					mCommandQueue.push(command);
				}
			}

			ImGui::EndDragDropTarget();
		}
	}

	// draw files in right pane
	for (size_t i = 0; i < filePaths.size(); i++)
	{
		if (ImGui::Selectable(fileLabels[i].c_str(), mHighlightedPath.compare(filePaths[i]) == 0, ImGuiSelectableFlags_AllowDoubleClick))
		{
			mHighlightedType = InteractionType::File;
			mHighlightedPath = filePaths[i];

			if (ImGui::IsMouseDoubleClicked(0))
			{
				if (fileTypes[i] == InteractionType::Scene)
				{
					ProjectDatabase::openScene(clipboard, filePaths[i]);
				}

				mSelectedFilePath = filePaths[i];

				clipboard.mSelectedType = fileTypes[i];
				clipboard.mSelectedPath = filePaths[i];
				clipboard.mSelectedId = ProjectDatabase::getGuid(clipboard.mSelectedPath);

				mFilter.Clear();
			}
		}

		if (ImGui::IsItemHovered())
		{
			mHoveredPath = filePaths[i];
		}

		if (ImGui::BeginDragDropSource())
		{
			std::string filePath = filePaths[i].string();
			const void* data = static_cast<const void*>(filePath.c_str());
			ImGui::SetDragDropPayload(InteractionTypeToDragAndDropTypeString(fileTypes[i]), data, filePath.length() + 1);
			ImGui::Text(filePath.c_str());
			ImGui::EndDragDropSource();
		}
	}

	if (!mSelectedDirectoryPath.empty())
	{
		drawPopupMenu(clipboard);
	}
}

void ProjectView::drawProjectNodeRecursive(ProjectNode* node)
{
	if (node != nullptr)
	{
		ImGuiTreeNodeFlags node_flags = ImGuiTreeNodeFlags_OpenOnArrow | ImGuiTreeNodeFlags_SpanFullWidth;
		if (node->mChildren.size() == 0)
		{
			node_flags |= ImGuiTreeNodeFlags_Leaf;
		}

		if (mSelectedDirectoryPath.compare(node->mDirectoryPath) == 0)
		{
			node_flags |= ImGuiTreeNodeFlags_Selected;
		}

		bool open = ImGui::TreeNodeEx(node->getDirectoryLabel().c_str(), node_flags);

		if (ImGui::BeginDragDropSource())
		{
			std::string directoryPath = node->mDirectoryPath.string();

			const void* data = static_cast<const void*>(directoryPath.c_str());

			ImGui::SetDragDropPayload(DragDropTypesToString(DragDropType::Folder), data, directoryPath.length() + 1);
			ImGui::Text(directoryPath.c_str());
			ImGui::EndDragDropSource();
		}

		if (ImGui::BeginDragDropTarget())
		{
			for (int i = 0; i < (int)DragDropType::Count; i++)
			{
				const ImGuiPayload* payload = ImGui::AcceptDragDropPayload(DragDropTypesToString(static_cast<DragDropType>(i)));
				if (payload != nullptr)
				{
					const char* data = static_cast<const char*>(payload->Data);

					Command command;
					command.mSource = std::string(data);
					command.mTarget = node->mDirectoryPath;
					command.mDragDropType = static_cast<DragDropType>(i);

					mCommandQueue.push(command);
				}
			}

			ImGui::EndDragDropTarget();
		}

		if (ImGui::IsItemHovered())
		{
			if (ImGui::IsMouseReleased(ImGuiMouseButton_::ImGuiMouseButton_Left))
			{
				mHighlightedType = InteractionType::Folder;
				mHighlightedPath = node->mDirectoryPath;
				mSelectedDirectoryPath = node->mDirectoryPath;
				mFilter.Clear();
			}
		}

		if (open)
		{
			// recurse for each sub directory
			for (size_t i = 0; i < node->mChildren.size(); i++)
			{
				drawProjectNodeRecursive(node->mChildren[i]);
			}

			ImGui::TreePop();
		}
	}
}

void ProjectView::drawPopupMenu(Clipboard& clipboard)
{
	// Right click popup menu
	if (ImGui::BeginPopupContextWindow("RightMouseClickPopup"))
	{
		if (ImGui::BeginMenu("Create..."))
		{
			if (ImGui::MenuItem("Folder"))
			{
				ProjectDatabase::createDirectory(mSelectedDirectoryPath);
			}

			ImGui::Separator();

			if (ImGui::BeginMenu("Shader..."))
			{
				if (ImGui::MenuItem("GLSL"))
				{
					ProjectDatabase::createShaderFile(mSelectedDirectoryPath);
				}

				ImGui::EndMenu();
			}

			if (ImGui::MenuItem("Cubemap"))
			{
				ProjectDatabase::createCubemapFile(clipboard.getWorld(), mSelectedDirectoryPath);
			}

			if (ImGui::MenuItem("Material"))
			{
				ProjectDatabase::createMaterialFile(clipboard.getWorld(), mSelectedDirectoryPath);
			}

			if (ImGui::MenuItem("RenderTexture"))
			{
				ProjectDatabase::createRenderTextureFile(clipboard.getWorld(), mSelectedDirectoryPath);
			}

			ImGui::EndMenu();
		}

		ImGui::EndPopup();
	}
}