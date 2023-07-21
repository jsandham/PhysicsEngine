#ifndef PREFERENCES_WINDOW_H__
#define PREFERENCES_WINDOW_H__

#include "../EditorClipboard.h"
#include "imgui.h"

namespace ImGui
{
	void StyleColorsDracula(ImGuiStyle* dst = NULL);
	void StyleColorsCherry(ImGuiStyle* dst = NULL);
	void StyleColorsLightGreen(ImGuiStyle* dst = NULL);
	void StyleColorsYellow(ImGuiStyle* dst = NULL);
	void StyleColorsGrey(ImGuiStyle* dst = NULL);
	void StyleColorsCharcoal(ImGuiStyle* dst = NULL);
	void StyleColorsCorporate(ImGuiStyle* dst = NULL);
	void StyleColorsCinder(ImGuiStyle* dst = NULL);
	void StyleColorsEnemyMouse(ImGuiStyle* dst = NULL);
	void StyleColorsDougBlinks(ImGuiStyle* dst = NULL);
	void StyleColorsGreenBlue(ImGuiStyle* dst = NULL);
	void StyleColorsRedDark(ImGuiStyle* dst = NULL);
	void StyleColorsDeepDark(ImGuiStyle* dst = NULL);

	void SetStyle(ImGuiStyle* src);

} // namespace ImGui

namespace PhysicsEditor
{
	enum class EditorStyle
	{
		Classic,
		Light,
		Dark,
		Dracula,
		Cherry,
		LightGreen,
		Yellow,
		Grey,
		Charcoal,
		Corporate,
		EnemyMouse,
		Cinder,
		DougBlinks,
		GreenBlue,
		RedDark,
		DeepDark,
		Count
	};

	constexpr auto EditorStyleToString(EditorStyle style)
	{
		switch (style)
		{
		case EditorStyle::Classic:
			return "Classic";
		case EditorStyle::Light:
			return "Light";
		case EditorStyle::Dark:
			return "Dark";
		case EditorStyle::Dracula:
			return "Dracula";
		case EditorStyle::Cherry:
			return "Cherry";
		case EditorStyle::LightGreen:
			return "LightGreen";
		case EditorStyle::Yellow:
			return "Yellow";
		case EditorStyle::Grey:
			return "Grey";
		case EditorStyle::Charcoal:
			return "Charcoal";
		case EditorStyle::Corporate:
			return "Corporate";
		case EditorStyle::EnemyMouse:
			return "EnemyMouse";
		case EditorStyle::Cinder:
			return "Cinder";
		case EditorStyle::DougBlinks:
			return "DougBlinks";
		case EditorStyle::GreenBlue:
			return "GreenBlue";
		case EditorStyle::RedDark:
			return "RedDark";
		case EditorStyle::DeepDark:
			return "DeepDark";
		}
	}

	class PreferencesWindow
	{
	private:
		EditorStyle mCurrentStyle;
		ImGuiStyle mStyles[EditorStyle::Count];
		bool mOpen;

	public:
		PreferencesWindow();
		~PreferencesWindow();
		PreferencesWindow(const PreferencesWindow& other) = delete;
		PreferencesWindow& operator=(const PreferencesWindow& other) = delete;

		void init(Clipboard& clipboard);
		void update(Clipboard& clipboard, bool isOpenedThisFrame);
	};
} // namespace PhysicsEditor

#endif
