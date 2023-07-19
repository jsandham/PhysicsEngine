#ifndef ABOUT_POPUP_H__
#define ABOUT_POPUP_H__

#include "../EditorClipboard.h"

namespace PhysicsEditor
{
	class AboutPopup
	{
	private:
		bool mOpen;

	public:
		AboutPopup();
		~AboutPopup();
		AboutPopup(const AboutPopup& other) = delete;
		AboutPopup& operator=(const AboutPopup& other) = delete;

		void init(Clipboard& clipboard);
		void update(Clipboard& clipboard, bool isOpenedThisFrame);
	};
} // namespace PhysicsEditor

#endif