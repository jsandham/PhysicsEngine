#ifndef PROJECT_VIEW_H__
#define PROJECT_VIEW_H__

#include <string>
#include <vector>
#include <filesystem>

#include "../EditorClipboard.h"

#include "imgui.h"

namespace PhysicsEditor
{
	enum class DragDropType
	{
		Folder,
		Material,
		Cubemap,
		Scene,
		Count
	};

	struct Command
	{
		std::filesystem::path mTarget; // mNewPath (always a directory)
		std::filesystem::path mSource; // mOldPath (can be directory or filepath)
		DragDropType mDragDropType;
	};

	struct ProjectNode
	{
		ProjectNode* mParent;
		std::vector<ProjectNode*> mChildren;

		std::filesystem::path mDirectoryPath;
		std::string mDirectoryName;
		std::string mDirectoryLabelEmpty;
		std::string mDirectoryLabelNonEmpty;
		std::vector<std::string> mFileNames;
		std::vector<std::string> mFileLabels;
		std::vector<InteractionType> mFileTypes;

		ProjectNode();
		ProjectNode(const std::filesystem::path& path);

		std::string getDirectoryLabel() const;
		std::filesystem::path getFilePath(size_t index) const;
		ProjectNode* addDirectory(const std::filesystem::path& path);
		void addFile(const std::filesystem::path& path);
	};

	struct ProjectTree
	{
		ProjectNode* mRoot;

		ProjectTree();
		~ProjectTree();

		void buildProjectTree(const std::filesystem::path& projectPath);
		void deleteProjectTree();
		void move(ProjectNode* target, ProjectNode* source);
		void move(ProjectNode* target, ProjectNode* source, const std::string& sourceFilename);
	};

	class ProjectView
	{
	private:
		ProjectTree mProjectTree;

		InteractionType mHighlightedType;

		std::filesystem::path mHighlightedPath;
		std::filesystem::path mHoveredPath;
		std::filesystem::path mSelectedDirectoryPath;
		std::filesystem::path mSelectedFilePath;

		ImGuiTextFilter mFilter;

		std::queue<Command> mCommandQueue;

		bool mBuildRequired;
		bool mOpen;

	public:
		ProjectView();
		~ProjectView();
		ProjectView(const ProjectView& other) = delete;
		ProjectView& operator=(const ProjectView& other) = delete;

		void init(Clipboard& clipboard);
		void update(Clipboard& clipboard, bool isOpenedThisFrame);

	private:
		void executeCommands();
		void drawLeftPane();
		void drawRightPane(Clipboard& clipboard);
		void drawProjectNodeRecursive(ProjectNode* node);
		void drawPopupMenu(Clipboard& clipboard);
	};
} // namespace PhysicsEditor

#endif
