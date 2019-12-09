#ifndef __EDITOR_UI_H__
#define __EDITOR_UI_H__

#include "core/Guid.h"

namespace PhysicsEditor
{
	/*enum DraggedType
	{
		DraggedEntity,
		DraggedComponent,
		DraggedSystem,
		DraggedFile,
		DraggedDirectory
	};*/

	struct EditorUI
	{
		PhysicsEngine::Guid draggedId;
		//DraggedType type;

	};
}

#endif
