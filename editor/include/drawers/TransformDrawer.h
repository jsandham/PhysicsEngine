#ifndef TRANSFORM_DRAWER_H__
#define TRANSFORM_DRAWER_H__

#include <imgui.h>
#include "../EditorClipboard.h"

namespace PhysicsEditor
{
	class TransformDrawer
	{
	private:
		ImVec2 mContentMin;
		ImVec2 mContentMax;

	public:
		TransformDrawer();
		~TransformDrawer();

		void render(Clipboard& clipboard, const PhysicsEngine::Guid& id);

	private:
		bool isHovered() const;
	};
} // namespace PhysicsEditor

#endif