#ifndef __MAIN_TOOLBAR_H__
#define __MAIN_TOOLBAR_H__

#include "EditorClipboard.h"

namespace PhysicsEditor
{
	class EditorToolbar 
	{
		public:
			EditorToolbar();
			~EditorToolbar();

			void render(EditorClipboard& clipboard);
	};
}

#endif
