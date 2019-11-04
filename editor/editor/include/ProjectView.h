#ifndef __PROJECT_H__
#define __PROJECT_H__

#include <string>
#include <vector>

namespace PhysicsEditor
{
	struct ProjectNode
	{
		ProjectNode* parent;
		std::vector<ProjectNode*> children;
		std::string directoryName;
		std::string directoryPath;
		bool isExpanded;
	};


	class ProjectView
	{
		private:
			std::string selectedNodeDirectoryPath;
			bool prevEditorApplicationActive;

		public:
			ProjectView();
			~ProjectView();

			void render(std::string currentProjectPath, bool editorApplicationActive, bool isOpenedThisFrame);


			void rebuildProjectTree();
			void drawProjectNodeRecursive(ProjectNode* node);
	};
}

#endif
