#ifndef __MAIN_TOOLBAR_H__
#define __MAIN_TOOLBAR_H__

#include "EditorUI.h"

namespace PhysicsEditor
{
	class EditorToolbar 
	{
		public:
			EditorToolbar();
			~EditorToolbar();

			void render(EditorUI ui);
	};
}

#endif
