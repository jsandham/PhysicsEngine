#ifndef __EDITOR_PROJECT_H__
#define __EDITOR_PROJECT_H__

namespace PhysicsEditor
{
	struct EditorProject
	{
		std::string name;
		std::string path;
		bool isDirty;
	};
}

#endif
