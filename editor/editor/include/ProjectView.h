#ifndef __PROJECT_H__
#define __PROJECT_H__

#include <string>
#include <vector>

namespace PhysicsEditor
{
	struct ProjectNode
	{
		std::vector<ProjectNode*> children;
		std::string directoryName;
		std::string directoryPath;
		bool isExpanded;
	};


	class ProjectView
	{
		public:
			ProjectView();
			~ProjectView();

			void render(std::string currentProjectPath, bool isOpenedThisFrame);

			void drawProjectNodeRecursive(ProjectNode* node);
	};
}

#endif
