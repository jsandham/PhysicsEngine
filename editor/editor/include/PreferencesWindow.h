#ifndef __PREFERENCES_WINDOW_H__
#define __PREFERENCES_WINDOW_H__

namespace PhysicsEditor
{
	class PreferencesWindow
	{
		private:
			bool isVisible;

		public:
			PreferencesWindow();
			~PreferencesWindow();

			void render(bool becomeVisibleThisFrame);
	};
}

#endif
