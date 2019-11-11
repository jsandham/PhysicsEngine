#ifndef __PROJECT_H__
#define __PROJECT_H__

#include <string>
#include <vector>

namespace PhysicsEditor
{
	struct ProjectNode
	{
		int id;
		ProjectNode* parent;
		std::vector<ProjectNode*> children;
		std::string directoryName;
		std::string directoryPath;
		bool isExpanded;

		ProjectNode() : id(-1), parent(NULL), directoryName(""), directoryPath(""), isExpanded(false){ }
	};


	class ProjectView
	{
		private:
			ProjectNode* root;
			ProjectNode* selected;
			std::vector<ProjectNode*> nodes;
			bool projectViewActive;

		public:
			ProjectView();
			~ProjectView();

			void render(std::string currentProjectPath, bool editorBecameActiveThisFrame, bool isOpenedThisFrame);

			void deleteProjectTree();
			void buildProjectTree(std::string currentProjectPath);
			void drawProjectTree();
			void drawProjectNodeRecursive(ProjectNode* node);
	};
}

#endif
