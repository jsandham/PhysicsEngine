#include "../../include/views/PreferencesWindow.h"
#include "../../include/IconsFontAwesome4.h"

using namespace PhysicsEditor;

static bool saveStyle(const char* filename, EditorStyle currentStyle, const ImGuiStyle* styles, int styleCount);
static bool loadStyle(const char* filename, EditorStyle* currentStyle, ImGuiStyle** styles, int styleCount);

PreferencesWindow::PreferencesWindow() : mOpen(true), mCurrentStyle(EditorStyle::Corporate)
{
}

PreferencesWindow::~PreferencesWindow()
{
}

void PreferencesWindow::init(Clipboard& clipboard)
{
	// Load styles save file
	ImGuiStyle* temp = &mStyles[0];
	if (!loadStyle("style_save_file.txt", &mCurrentStyle, &temp, (int)EditorStyle::Count))
	{
		// Cannot load from save file, use default styles instead
		ImGui::StyleColorsClassic(&mStyles[(int)EditorStyle::Classic]);
		ImGui::StyleColorsLight(&mStyles[(int)EditorStyle::Light]);
		ImGui::StyleColorsDark(&mStyles[(int)EditorStyle::Dark]);
		ImGui::StyleColorsDracula(&mStyles[(int)EditorStyle::Dracula]);
		ImGui::StyleColorsCherry(&mStyles[(int)EditorStyle::Cherry]);
		ImGui::StyleColorsLightGreen(&mStyles[(int)EditorStyle::LightGreen]);
		ImGui::StyleColorsYellow(&mStyles[(int)EditorStyle::Yellow]);
		ImGui::StyleColorsGrey(&mStyles[(int)EditorStyle::Grey]);
		ImGui::StyleColorsCharcoal(&mStyles[(int)EditorStyle::Charcoal]);
		ImGui::StyleColorsCorporate(&mStyles[(int)EditorStyle::Corporate]);
		ImGui::StyleColorsCinder(&mStyles[(int)EditorStyle::Cinder]);
		ImGui::StyleColorsEnemyMouse(&mStyles[(int)EditorStyle::EnemyMouse]);
		ImGui::StyleColorsDougBlinks(&mStyles[(int)EditorStyle::DougBlinks]);
		ImGui::StyleColorsGreenBlue(&mStyles[(int)EditorStyle::GreenBlue]);
		ImGui::StyleColorsRedDark(&mStyles[(int)EditorStyle::RedDark]);
		ImGui::StyleColorsDeepDark(&mStyles[(int)EditorStyle::DeepDark]);

		mCurrentStyle = EditorStyle::Corporate;
	}

	ImGui::SetStyle(&mStyles[(int)mCurrentStyle]);

	ImFontConfig config;
	config.MergeMode = true;
	config.GlyphMinAdvanceX = 13.0f; // Use if you want to make the icon monospaced
	static const ImWchar icon_ranges[] = { ICON_MIN_FA, ICON_MAX_FA, 0 };

	// Load fonts
	ImGuiIO& io = ImGui::GetIO();
	io.Fonts->AddFontDefault();
	io.Fonts->AddFontFromFileTTF("fonts/fontawesome-webfont.ttf", 13.0f, &config, icon_ranges);
	io.Fonts->AddFontFromFileTTF("fonts/ProggyTiny.ttf", 10.0f);
	io.Fonts->AddFontFromFileTTF("fonts/fontawesome-webfont.ttf", 13.0f, &config, icon_ranges);
	io.Fonts->AddFontFromFileTTF("fonts/Cousine-Regular.ttf", 13.0f);
	io.Fonts->AddFontFromFileTTF("fonts/fontawesome-webfont.ttf", 13.0f, &config, icon_ranges);
	io.Fonts->AddFontFromFileTTF("fonts/DroidSans.ttf", 13.0f);
	io.Fonts->AddFontFromFileTTF("fonts/fontawesome-webfont.ttf", 13.0f, &config, icon_ranges);
	io.Fonts->AddFontFromFileTTF("fonts/Karla-Regular.ttf", 13.0f);
	io.Fonts->AddFontFromFileTTF("fonts/fontawesome-webfont.ttf", 13.0f, &config, icon_ranges);

	io.Fonts->Build();
}

static void HelpMarker(const char* desc)
{
	ImGui::TextDisabled("(?)");
	if (ImGui::BeginItemTooltip())
	{
		ImGui::PushTextWrapPos(ImGui::GetFontSize() * 35.0f);
		ImGui::TextUnformatted(desc);
		ImGui::PopTextWrapPos();
		ImGui::EndTooltip();
	}
}

void PreferencesWindow::update(Clipboard& clipboard, bool isOpenedThisFrame)
{
	if (isOpenedThisFrame)
	{
		ImGui::SetNextWindowPos(ImVec2(400.0f, 100.0f));
		ImGui::SetNextWindowSize(ImVec2(600.0f, 380.0f));

		ImGui::OpenPopup("##Preferences...");
		mOpen = true;
	}

	if (ImGui::BeginPopupModal("##Preferences...", &mOpen, ImGuiWindowFlags_NoResize))
	{
		// Order matters here
		const char* themeNames[] = { "Classic",    "Light",  "Dark", "Dracula",  "Cherry",
									"LightGreen", "Yellow", "Grey", "Charcoal", "Corporate",
									"EnemyMouse", "Cinder", "DougBlinks", "GreenBlue", "RedDark", "DeepDark" };

		int index = static_cast<int>(mCurrentStyle);
		if (ImGui::BeginCombo("Themes", themeNames[index]))
		{
			for (int n = 0; n < static_cast<int>(EditorStyle::Count); n++)
			{
				bool is_selected = false;
				if (ImGui::Selectable(themeNames[n], is_selected))
				{
					mCurrentStyle = static_cast<EditorStyle>(n);
					ImGui::SetStyle(&mStyles[(int)mCurrentStyle]);

					index = n;
				}
			}
			ImGui::EndCombo();
		}

		ImGuiIO& io = ImGui::GetIO();
		ImFont* font_current = ImGui::GetFont();
		if (ImGui::BeginCombo("Fonts", font_current->GetDebugName()))
		{
			for (int n = 0; n < io.Fonts->Fonts.Size; n++)
			{
				ImFont* font = io.Fonts->Fonts[n];
				ImGui::PushID((void*)font);
				if (ImGui::Selectable(font->GetDebugName(), font == font_current))
					io.FontDefault = font;
				ImGui::PopID();
			}
			ImGui::EndCombo();
		}

		ImGuiStyle& style = mStyles[(int)mCurrentStyle];

		ImGui::Separator();

		if (ImGui::BeginTabBar("##tabs", ImGuiTabBarFlags_None))
		{
			if (ImGui::BeginTabItem("Padding"))
			{
				ImGui::SliderFloat2("WindowPadding", (float*)&style.WindowPadding, 0.0f, 20.0f, "%.0f");
				ImGui::SliderFloat2("FramePadding", (float*)&style.FramePadding, 0.0f, 20.0f, "%.0f");
				ImGui::SliderFloat2("CellPadding", (float*)&style.CellPadding, 0.0f, 20.0f, "%.0f");
				ImGui::SliderFloat2("ItemSpacing", (float*)&style.ItemSpacing, 0.0f, 20.0f, "%.0f");
				ImGui::SliderFloat2("ItemInnerSpacing", (float*)&style.ItemInnerSpacing, 0.0f, 20.0f, "%.0f");
				ImGui::SliderFloat2("TouchExtraPadding", (float*)&style.TouchExtraPadding, 0.0f, 10.0f, "%.0f");
				ImGui::SliderFloat("IndentSpacing", &style.IndentSpacing, 0.0f, 30.0f, "%.0f");
				ImGui::SliderFloat("ScrollbarSize", &style.ScrollbarSize, 1.0f, 20.0f, "%.0f");
				ImGui::SliderFloat("GrabMinSize", &style.GrabMinSize, 1.0f, 20.0f, "%.0f");

				ImGui::SliderFloat2("DisplaySafeAreaPadding", (float*)&style.DisplaySafeAreaPadding, 0.0f, 30.0f, "%.0f"); ImGui::SameLine(); HelpMarker("Adjust if you cannot see the edges of your screen (e.g. on a TV where scaling has not been configured).");

				ImGui::EndTabItem();
			}

			if (ImGui::BeginTabItem("Borders"))
			{
				ImGui::SliderFloat("WindowBorderSize", &style.WindowBorderSize, 0.0f, 1.0f, "%.0f");
				ImGui::SliderFloat("ChildBorderSize", &style.ChildBorderSize, 0.0f, 1.0f, "%.0f");
				ImGui::SliderFloat("PopupBorderSize", &style.PopupBorderSize, 0.0f, 1.0f, "%.0f");
				ImGui::SliderFloat("FrameBorderSize", &style.FrameBorderSize, 0.0f, 1.0f, "%.0f");
				ImGui::SliderFloat("TabBorderSize", &style.TabBorderSize, 0.0f, 1.0f, "%.0f");

				ImGui::EndTabItem();
			}

			if (ImGui::BeginTabItem("Rounding"))
			{
				ImGui::SliderFloat("WindowRounding", &style.WindowRounding, 0.0f, 12.0f, "%.0f");
				ImGui::SliderFloat("ChildRounding", &style.ChildRounding, 0.0f, 12.0f, "%.0f");
				ImGui::SliderFloat("FrameRounding", &style.FrameRounding, 0.0f, 12.0f, "%.0f");
				ImGui::SliderFloat("PopupRounding", &style.PopupRounding, 0.0f, 12.0f, "%.0f");
				ImGui::SliderFloat("ScrollbarRounding", &style.ScrollbarRounding, 0.0f, 12.0f, "%.0f");
				ImGui::SliderFloat("GrabRounding", &style.GrabRounding, 0.0f, 12.0f, "%.0f");
				ImGui::SliderFloat("TabRounding", &style.TabRounding, 0.0f, 12.0f, "%.0f");

				ImGui::EndTabItem();
			}

			if (ImGui::BeginTabItem("Widgets"))
			{
				ImGui::SliderFloat2("WindowTitleAlign", (float*)&style.WindowTitleAlign, 0.0f, 1.0f, "%.2f");
				int window_menu_button_position = style.WindowMenuButtonPosition + 1;
				if (ImGui::Combo("WindowMenuButtonPosition", (int*)&window_menu_button_position, "None\0Left\0Right\0"))
					style.WindowMenuButtonPosition = window_menu_button_position - 1;
				ImGui::Combo("ColorButtonPosition", (int*)&style.ColorButtonPosition, "Left\0Right\0");
				ImGui::SliderFloat2("ButtonTextAlign", (float*)&style.ButtonTextAlign, 0.0f, 1.0f, "%.2f");
				ImGui::SameLine(); HelpMarker("Alignment applies when a button is larger than its text content.");
				ImGui::SliderFloat2("SelectableTextAlign", (float*)&style.SelectableTextAlign, 0.0f, 1.0f, "%.2f");
				ImGui::SameLine(); HelpMarker("Alignment applies when a selectable is larger than its text content.");
				ImGui::SliderFloat("SeparatorTextBorderSize", &style.SeparatorTextBorderSize, 0.0f, 10.0f, "%.0f");
				ImGui::SliderFloat2("SeparatorTextAlign", (float*)&style.SeparatorTextAlign, 0.0f, 1.0f, "%.2f");
				ImGui::SliderFloat2("SeparatorTextPadding", (float*)&style.SeparatorTextPadding, 0.0f, 40.0f, "%.0f");
				ImGui::SliderFloat("LogSliderDeadzone", &style.LogSliderDeadzone, 0.0f, 12.0f, "%.0f");

				ImGui::EndTabItem();
			}

			if (ImGui::BeginTabItem("Tooltips"))
			{
				for (int n = 0; n < 2; n++)
				{
					if (ImGui::TreeNodeEx(n == 0 ? "HoverFlagsForTooltipMouse" : "HoverFlagsForTooltipNav"))
					{
						ImGuiHoveredFlags* p = (n == 0) ? &style.HoverFlagsForTooltipMouse : &style.HoverFlagsForTooltipNav;
						ImGui::CheckboxFlags("ImGuiHoveredFlags_DelayNone", p, ImGuiHoveredFlags_DelayNone);
						ImGui::CheckboxFlags("ImGuiHoveredFlags_DelayShort", p, ImGuiHoveredFlags_DelayShort);
						ImGui::CheckboxFlags("ImGuiHoveredFlags_DelayNormal", p, ImGuiHoveredFlags_DelayNormal);
						ImGui::CheckboxFlags("ImGuiHoveredFlags_Stationary", p, ImGuiHoveredFlags_Stationary);
						ImGui::CheckboxFlags("ImGuiHoveredFlags_NoSharedDelay", p, ImGuiHoveredFlags_NoSharedDelay);
						ImGui::TreePop();
					}
				}

				ImGui::EndTabItem();
			}

			if (ImGui::BeginTabItem("Rendering"))
			{
				ImGui::Checkbox("Anti-aliased lines", &style.AntiAliasedLines);
				ImGui::SameLine();
				HelpMarker("When disabling anti-aliasing lines, you'll probably want to disable borders in your style as well.");

				ImGui::Checkbox("Anti-aliased lines use texture", &style.AntiAliasedLinesUseTex);
				ImGui::SameLine();
				HelpMarker("Faster lines using texture data. Require backend to render with bilinear filtering (not point/nearest filtering).");

				ImGui::Checkbox("Anti-aliased fill", &style.AntiAliasedFill);
				ImGui::PushItemWidth(ImGui::GetFontSize() * 8);
				ImGui::DragFloat("Curve Tessellation Tolerance", &style.CurveTessellationTol, 0.02f, 0.10f, 10.0f, "%.2f");
				if (style.CurveTessellationTol < 0.10f) style.CurveTessellationTol = 0.10f;

				// When editing the "Circle Segment Max Error" value, draw a preview of its effect on auto-tessellated circles.
				ImGui::DragFloat("Circle Tessellation Max Error", &style.CircleTessellationMaxError, 0.005f, 0.10f, 5.0f, "%.2f", ImGuiSliderFlags_AlwaysClamp);
				const bool show_samples = ImGui::IsItemActive();
				if (show_samples)
					ImGui::SetNextWindowPos(ImGui::GetCursorScreenPos());
				if (show_samples && ImGui::BeginTooltip())
				{
					ImGui::TextUnformatted("(R = radius, N = number of segments)");
					ImGui::Spacing();
					ImDrawList* draw_list = ImGui::GetWindowDrawList();
					const float min_widget_width = ImGui::CalcTextSize("N: MMM\nR: MMM").x;
					for (int n = 0; n < 8; n++)
					{
						const float RAD_MIN = 5.0f;
						const float RAD_MAX = 70.0f;
						const float rad = RAD_MIN + (RAD_MAX - RAD_MIN) * (float)n / (8.0f - 1.0f);

						ImGui::BeginGroup();

						ImGui::Text("R: %.f\nN: %d", rad, draw_list->_CalcCircleAutoSegmentCount(rad));

						const float canvas_width = std::max(min_widget_width, rad * 2.0f);
						const float offset_x = floorf(canvas_width * 0.5f);
						const float offset_y = floorf(RAD_MAX);

						const ImVec2 p1 = ImGui::GetCursorScreenPos();
						draw_list->AddCircle(ImVec2(p1.x + offset_x, p1.y + offset_y), rad, ImGui::GetColorU32(ImGuiCol_Text));
						ImGui::Dummy(ImVec2(canvas_width, RAD_MAX * 2));

						ImGui::EndGroup();
						ImGui::SameLine();
					}
					ImGui::EndTooltip();
				}
				ImGui::SameLine();
				HelpMarker("When drawing circle primitives with \"num_segments == 0\" tesselation will be calculated automatically.");

				ImGui::DragFloat("Global Alpha", &style.Alpha, 0.005f, 0.20f, 1.0f, "%.2f"); // Not exposing zero here so user doesn't "lose" the UI (zero alpha clips all widgets). But application code could have a toggle to switch between zero and non-zero.
				ImGui::DragFloat("Disabled Alpha", &style.DisabledAlpha, 0.005f, 0.0f, 1.0f, "%.2f"); ImGui::SameLine(); HelpMarker("Additional alpha multiplier for disabled items (multiply over current value of Alpha).");
				ImGui::PopItemWidth();

				ImGui::EndTabItem();
			}

			ImGui::EndTabBar();
		}

		ImGui::SetStyle(&mStyles[(int)mCurrentStyle]);

		if (ImGui::Button("Ok"))
		{
			saveStyle("style_save_file.txt", mCurrentStyle, &mStyles[0], (int)EditorStyle::Count);
			ImGui::CloseCurrentPopup();
		}

		ImGui::SameLine();

		if (ImGui::Button("Reset to default"))
		{
			switch (mCurrentStyle)
			{
			case EditorStyle::Classic: { ImGui::StyleColorsClassic(&mStyles[(int)mCurrentStyle]); break; }
			case EditorStyle::Light: { ImGui::StyleColorsLight(&mStyles[(int)mCurrentStyle]); break; }
			case EditorStyle::Dark: { ImGui::StyleColorsDark(&mStyles[(int)mCurrentStyle]); break; }
			case EditorStyle::Dracula: { ImGui::StyleColorsDracula(&mStyles[(int)mCurrentStyle]); break; }
			case EditorStyle::Cherry: { ImGui::StyleColorsCherry(&mStyles[(int)mCurrentStyle]); break; }
			case EditorStyle::LightGreen: { ImGui::StyleColorsLightGreen(&mStyles[(int)mCurrentStyle]); break; }
			case EditorStyle::Yellow: { ImGui::StyleColorsYellow(&mStyles[(int)mCurrentStyle]); break; }
			case EditorStyle::Grey: { ImGui::StyleColorsGrey(&mStyles[(int)mCurrentStyle]); break; }
			case EditorStyle::Charcoal: { ImGui::StyleColorsCharcoal(&mStyles[(int)mCurrentStyle]); break; }
			case EditorStyle::Corporate: { ImGui::StyleColorsCorporate(&mStyles[(int)mCurrentStyle]); break; }
			case EditorStyle::EnemyMouse: { ImGui::StyleColorsEnemyMouse(&mStyles[(int)mCurrentStyle]); break; }
			case EditorStyle::Cinder: { ImGui::StyleColorsCinder(&mStyles[(int)mCurrentStyle]); break; }
			case EditorStyle::DougBlinks: { ImGui::StyleColorsDougBlinks(&mStyles[(int)mCurrentStyle]); break; }
			case EditorStyle::GreenBlue: { ImGui::StyleColorsGreenBlue(&mStyles[(int)mCurrentStyle]); break; }
			case EditorStyle::RedDark: { ImGui::StyleColorsRedDark(&mStyles[(int)mCurrentStyle]); break; }
			case EditorStyle::DeepDark: { ImGui::StyleColorsDeepDark(&mStyles[(int)mCurrentStyle]); break; }
			}

			ImGui::SetStyle(&mStyles[(int)mCurrentStyle]);
		}

		ImGui::EndPopup();
	}
}

// Styles
void ImGui::StyleColorsDracula(ImGuiStyle* dst)
{
	ImGuiStyle* style = dst ? dst : &ImGui::GetStyle();
	ImVec4* colors = style->Colors;

	style->WindowRounding = 5.3f;
	style->GrabRounding = style->FrameRounding = 2.3f;
	style->ScrollbarRounding = 5.0f;
	style->FrameBorderSize = 1.0f;
	style->ItemSpacing.y = 6.5f;

	colors[ImGuiCol_Text] = { 0.73333335f, 0.73333335f, 0.73333335f, 1.00f };
	colors[ImGuiCol_TextDisabled] = { 0.34509805f, 0.34509805f, 0.34509805f, 1.00f };
	colors[ImGuiCol_WindowBg] = { 0.23529413f, 0.24705884f, 0.25490198f, 0.94f };
	colors[ImGuiCol_ChildBg] = { 0.23529413f, 0.24705884f, 0.25490198f, 0.00f };
	colors[ImGuiCol_PopupBg] = { 0.23529413f, 0.24705884f, 0.25490198f, 0.94f };
	colors[ImGuiCol_Border] = { 0.33333334f, 0.33333334f, 0.33333334f, 0.50f };
	colors[ImGuiCol_BorderShadow] = { 0.15686275f, 0.15686275f, 0.15686275f, 0.00f };
	colors[ImGuiCol_FrameBg] = { 0.16862746f, 0.16862746f, 0.16862746f, 0.54f };
	colors[ImGuiCol_FrameBgHovered] = { 0.453125f, 0.67578125f, 0.99609375f, 0.67f };
	colors[ImGuiCol_FrameBgActive] = { 0.47058827f, 0.47058827f, 0.47058827f, 0.67f };
	colors[ImGuiCol_TitleBg] = { 0.04f, 0.04f, 0.04f, 1.00f };
	colors[ImGuiCol_TitleBgCollapsed] = { 0.16f, 0.29f, 0.48f, 1.00f };
	colors[ImGuiCol_TitleBgActive] = { 0.00f, 0.00f, 0.00f, 0.51f };
	colors[ImGuiCol_MenuBarBg] = { 0.27058825f, 0.28627452f, 0.2901961f, 0.80f };
	colors[ImGuiCol_ScrollbarBg] = { 0.27058825f, 0.28627452f, 0.2901961f, 0.60f };
	colors[ImGuiCol_ScrollbarGrab] = { 0.21960786f, 0.30980393f, 0.41960788f, 0.51f };
	colors[ImGuiCol_ScrollbarGrabHovered] = { 0.21960786f, 0.30980393f, 0.41960788f, 1.00f };
	colors[ImGuiCol_ScrollbarGrabActive] = { 0.13725491f, 0.19215688f, 0.2627451f, 0.91f };
	// colors[ImGuiCol_ComboBg]               = {0.1f, 0.1f, 0.1f, 0.99f};
	colors[ImGuiCol_CheckMark] = { 0.90f, 0.90f, 0.90f, 0.83f };
	colors[ImGuiCol_SliderGrab] = { 0.70f, 0.70f, 0.70f, 0.62f };
	colors[ImGuiCol_SliderGrabActive] = { 0.30f, 0.30f, 0.30f, 0.84f };
	colors[ImGuiCol_Button] = { 0.33333334f, 0.3529412f, 0.36078432f, 0.49f };
	colors[ImGuiCol_ButtonHovered] = { 0.21960786f, 0.30980393f, 0.41960788f, 1.00f };
	colors[ImGuiCol_ButtonActive] = { 0.13725491f, 0.19215688f, 0.2627451f, 1.00f };
	colors[ImGuiCol_Header] = { 0.33333334f, 0.3529412f, 0.36078432f, 0.53f };
	colors[ImGuiCol_HeaderHovered] = { 0.453125f, 0.67578125f, 0.99609375f, 0.67f };
	colors[ImGuiCol_HeaderActive] = { 0.47058827f, 0.47058827f, 0.47058827f, 0.67f };
	colors[ImGuiCol_Separator] = { 0.31640625f, 0.31640625f, 0.31640625f, 1.00f };
	colors[ImGuiCol_SeparatorHovered] = { 0.31640625f, 0.31640625f, 0.31640625f, 1.00f };
	colors[ImGuiCol_SeparatorActive] = { 0.31640625f, 0.31640625f, 0.31640625f, 1.00f };
	colors[ImGuiCol_ResizeGrip] = { 1.00f, 1.00f, 1.00f, 0.85f };
	colors[ImGuiCol_ResizeGripHovered] = { 1.00f, 1.00f, 1.00f, 0.60f };
	colors[ImGuiCol_ResizeGripActive] = { 1.00f, 1.00f, 1.00f, 0.90f };
	colors[ImGuiCol_PlotLines] = { 0.61f, 0.61f, 0.61f, 1.00f };
	colors[ImGuiCol_PlotLinesHovered] = { 1.00f, 0.43f, 0.35f, 1.00f };
	colors[ImGuiCol_PlotHistogram] = { 0.90f, 0.70f, 0.00f, 1.00f };
	colors[ImGuiCol_PlotHistogramHovered] = { 1.00f, 0.60f, 0.00f, 1.00f };
	colors[ImGuiCol_TextSelectedBg] = { 0.18431373f, 0.39607847f, 0.79215693f, 0.90f };
}

void ImGui::StyleColorsCherry(ImGuiStyle* dst)
{
	ImGuiStyle* style = dst ? dst : &ImGui::GetStyle();
	ImVec4* colors = style->Colors;

	// cherry colors, 3 intensities
#define HI(v) ImVec4(0.502f, 0.075f, 0.256f, v)
#define MED(v) ImVec4(0.455f, 0.198f, 0.301f, v)
#define LOW(v) ImVec4(0.232f, 0.201f, 0.271f, v)
// backgrounds (@todo: complete with BG_MED, BG_LOW)
#define BG(v) ImVec4(0.200f, 0.220f, 0.270f, v)
// text
#define TEXT(v) ImVec4(0.860f, 0.930f, 0.890f, v)

	colors[ImGuiCol_Text] = TEXT(0.78f);
	colors[ImGuiCol_TextDisabled] = TEXT(0.28f);
	colors[ImGuiCol_WindowBg] = ImVec4(0.13f, 0.14f, 0.17f, 1.00f);
	colors[ImGuiCol_PopupBg] = BG(0.9f);
	colors[ImGuiCol_Border] = ImVec4(0.31f, 0.31f, 1.00f, 0.00f);
	colors[ImGuiCol_BorderShadow] = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
	colors[ImGuiCol_FrameBg] = BG(1.00f);
	colors[ImGuiCol_FrameBgHovered] = MED(0.78f);
	colors[ImGuiCol_FrameBgActive] = MED(1.00f);
	colors[ImGuiCol_TitleBg] = LOW(1.00f);
	colors[ImGuiCol_TitleBgActive] = HI(1.00f);
	colors[ImGuiCol_TitleBgCollapsed] = BG(0.75f);
	colors[ImGuiCol_MenuBarBg] = BG(0.47f);
	colors[ImGuiCol_ScrollbarBg] = BG(1.00f);
	colors[ImGuiCol_ScrollbarGrab] = ImVec4(0.09f, 0.15f, 0.16f, 1.00f);
	colors[ImGuiCol_ScrollbarGrabHovered] = MED(0.78f);
	colors[ImGuiCol_ScrollbarGrabActive] = MED(1.00f);
	colors[ImGuiCol_CheckMark] = ImVec4(0.71f, 0.22f, 0.27f, 1.00f);
	colors[ImGuiCol_SliderGrab] = ImVec4(0.47f, 0.77f, 0.83f, 0.14f);
	colors[ImGuiCol_SliderGrabActive] = ImVec4(0.71f, 0.22f, 0.27f, 1.00f);
	colors[ImGuiCol_Button] = ImVec4(0.47f, 0.77f, 0.83f, 0.14f);
	colors[ImGuiCol_ButtonHovered] = MED(0.86f);
	colors[ImGuiCol_ButtonActive] = MED(1.00f);
	colors[ImGuiCol_Header] = MED(0.76f);
	colors[ImGuiCol_HeaderHovered] = MED(0.86f);
	colors[ImGuiCol_HeaderActive] = HI(1.00f);
	// colors[ImGuiCol_Column] = ImVec4(0.14f, 0.16f, 0.19f, 1.00f);
	// colors[ImGuiCol_ColumnHovered] = MED(0.78f);
	// colors[ImGuiCol_ColumnActive] = MED(1.00f);
	colors[ImGuiCol_ResizeGrip] = ImVec4(0.47f, 0.77f, 0.83f, 0.04f);
	colors[ImGuiCol_ResizeGripHovered] = MED(0.78f);
	colors[ImGuiCol_ResizeGripActive] = MED(1.00f);
	colors[ImGuiCol_PlotLines] = TEXT(0.63f);
	colors[ImGuiCol_PlotLinesHovered] = MED(1.00f);
	colors[ImGuiCol_PlotHistogram] = TEXT(0.63f);
	colors[ImGuiCol_PlotHistogramHovered] = MED(1.00f);
	colors[ImGuiCol_TextSelectedBg] = MED(0.43f);
	// [...]
	//colors[ImGuiCol_ModalWindowDarkening] = BG(0.73f);

	style->WindowPadding = ImVec2(6, 4);
	style->WindowRounding = 0.0f;
	style->FramePadding = ImVec2(5, 2);
	style->FrameRounding = 3.0f;
	style->ItemSpacing = ImVec2(7, 1);
	style->ItemInnerSpacing = ImVec2(1, 1);
	style->TouchExtraPadding = ImVec2(0, 0);
	style->IndentSpacing = 6.0f;
	style->ScrollbarSize = 12.0f;
	style->ScrollbarRounding = 16.0f;
	style->GrabMinSize = 20.0f;
	style->GrabRounding = 2.0f;

	style->WindowTitleAlign.x = 0.50f;

	colors[ImGuiCol_Border] = ImVec4(0.539f, 0.479f, 0.255f, 0.162f);
	style->FrameBorderSize = 0.0f;
	style->WindowBorderSize = 1.0f;
}

void ImGui::StyleColorsLightGreen(ImGuiStyle* dst)
{
	ImGuiStyle* style = dst ? dst : &ImGui::GetStyle();
	ImVec4* colors = style->Colors;

	style->WindowRounding = 2.0f;    // Radius of window corners rounding. Set to 0.0f to have rectangular windows
	style->ScrollbarRounding = 3.0f; // Radius of grab corners rounding for scrollbar
	style->GrabRounding = 2.0f;      // Radius of grabs corners rounding. Set to 0.0f to have rectangular slider grabs.
	style->AntiAliasedLines = true;
	style->AntiAliasedFill = true;
	style->WindowRounding = 2;
	style->ChildRounding = 2;
	style->ScrollbarSize = 16;
	style->ScrollbarRounding = 3;
	style->GrabRounding = 2;
	style->ItemSpacing.x = 10;
	style->ItemSpacing.y = 4;
	style->IndentSpacing = 22;
	style->FramePadding.x = 6;
	style->FramePadding.y = 4;
	style->Alpha = 1.0f;
	style->FrameRounding = 3.0f;

	colors[ImGuiCol_Text] = ImVec4(0.00f, 0.00f, 0.00f, 1.00f);
	colors[ImGuiCol_TextDisabled] = ImVec4(0.60f, 0.60f, 0.60f, 1.00f);
	colors[ImGuiCol_WindowBg] = ImVec4(0.86f, 0.86f, 0.86f, 1.00f);
	// colors[ImGuiCol_ChildWindowBg]         = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
	colors[ImGuiCol_ChildBg] = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
	colors[ImGuiCol_PopupBg] = ImVec4(0.93f, 0.93f, 0.93f, 0.98f);
	colors[ImGuiCol_Border] = ImVec4(0.71f, 0.71f, 0.71f, 0.08f);
	colors[ImGuiCol_BorderShadow] = ImVec4(0.00f, 0.00f, 0.00f, 0.04f);
	colors[ImGuiCol_FrameBg] = ImVec4(0.71f, 0.71f, 0.71f, 0.55f);
	colors[ImGuiCol_FrameBgHovered] = ImVec4(0.94f, 0.94f, 0.94f, 0.55f);
	colors[ImGuiCol_FrameBgActive] = ImVec4(0.71f, 0.78f, 0.69f, 0.98f);
	colors[ImGuiCol_TitleBg] = ImVec4(0.85f, 0.85f, 0.85f, 1.00f);
	colors[ImGuiCol_TitleBgCollapsed] = ImVec4(0.82f, 0.78f, 0.78f, 0.51f);
	colors[ImGuiCol_TitleBgActive] = ImVec4(0.78f, 0.78f, 0.78f, 1.00f);
	colors[ImGuiCol_MenuBarBg] = ImVec4(0.86f, 0.86f, 0.86f, 1.00f);
	colors[ImGuiCol_ScrollbarBg] = ImVec4(0.20f, 0.25f, 0.30f, 0.61f);
	colors[ImGuiCol_ScrollbarGrab] = ImVec4(0.90f, 0.90f, 0.90f, 0.30f);
	colors[ImGuiCol_ScrollbarGrabHovered] = ImVec4(0.92f, 0.92f, 0.92f, 0.78f);
	colors[ImGuiCol_ScrollbarGrabActive] = ImVec4(1.00f, 1.00f, 1.00f, 1.00f);
	colors[ImGuiCol_CheckMark] = ImVec4(0.184f, 0.407f, 0.193f, 1.00f);
	colors[ImGuiCol_SliderGrab] = ImVec4(0.26f, 0.59f, 0.98f, 0.78f);
	colors[ImGuiCol_SliderGrabActive] = ImVec4(0.26f, 0.59f, 0.98f, 1.00f);
	colors[ImGuiCol_Button] = ImVec4(0.71f, 0.78f, 0.69f, 0.40f);
	colors[ImGuiCol_ButtonHovered] = ImVec4(0.725f, 0.805f, 0.702f, 1.00f);
	colors[ImGuiCol_ButtonActive] = ImVec4(0.793f, 0.900f, 0.836f, 1.00f);
	colors[ImGuiCol_Header] = ImVec4(0.71f, 0.78f, 0.69f, 0.31f);
	colors[ImGuiCol_HeaderHovered] = ImVec4(0.71f, 0.78f, 0.69f, 0.80f);
	colors[ImGuiCol_HeaderActive] = ImVec4(0.71f, 0.78f, 0.69f, 1.00f);
	// colors[ImGuiCol_Column] = ImVec4(0.39f, 0.39f, 0.39f, 1.00f);
	// colors[ImGuiCol_ColumnHovered] = ImVec4(0.26f, 0.59f, 0.98f, 0.78f);
	// colors[ImGuiCol_ColumnActive] = ImVec4(0.26f, 0.59f, 0.98f, 1.00f);
	colors[ImGuiCol_Separator] = ImVec4(0.39f, 0.39f, 0.39f, 1.00f);
	colors[ImGuiCol_SeparatorHovered] = ImVec4(0.14f, 0.44f, 0.80f, 0.78f);
	colors[ImGuiCol_SeparatorActive] = ImVec4(0.14f, 0.44f, 0.80f, 1.00f);
	colors[ImGuiCol_ResizeGrip] = ImVec4(1.00f, 1.00f, 1.00f, 0.00f);
	colors[ImGuiCol_ResizeGripHovered] = ImVec4(0.26f, 0.59f, 0.98f, 0.45f);
	colors[ImGuiCol_ResizeGripActive] = ImVec4(0.26f, 0.59f, 0.98f, 0.78f);
	colors[ImGuiCol_PlotLines] = ImVec4(0.39f, 0.39f, 0.39f, 1.00f);
	colors[ImGuiCol_PlotLinesHovered] = ImVec4(1.00f, 0.43f, 0.35f, 1.00f);
	colors[ImGuiCol_PlotHistogram] = ImVec4(0.90f, 0.70f, 0.00f, 1.00f);
	colors[ImGuiCol_PlotHistogramHovered] = ImVec4(1.00f, 0.60f, 0.00f, 1.00f);
	colors[ImGuiCol_TextSelectedBg] = ImVec4(0.26f, 0.59f, 0.98f, 0.35f);
	//colors[ImGuiCol_ModalWindowDarkening] = ImVec4(0.20f, 0.20f, 0.20f, 0.35f);
	colors[ImGuiCol_DragDropTarget] = ImVec4(0.26f, 0.59f, 0.98f, 0.95f);
	colors[ImGuiCol_NavHighlight] = colors[ImGuiCol_HeaderHovered];
	colors[ImGuiCol_NavWindowingHighlight] = ImVec4(0.70f, 0.70f, 0.70f, 0.70f);
}

void ImGui::StyleColorsYellow(ImGuiStyle* dst)
{
	ImGuiStyle* style = dst ? dst : &ImGui::GetStyle();
	ImVec4* colors = style->Colors;

	style->WindowPadding = ImVec2(15, 15);
	style->WindowRounding = 5.0f;
	style->FramePadding = ImVec2(5, 5);
	style->FrameRounding = 4.0f;
	style->ItemSpacing = ImVec2(12, 8);
	style->ItemInnerSpacing = ImVec2(8, 6);
	style->IndentSpacing = 25.0f;
	style->ScrollbarSize = 15.0f;
	style->ScrollbarRounding = 9.0f;
	style->GrabMinSize = 5.0f;
	style->GrabRounding = 3.0f;

	colors[ImGuiCol_Text] = ImVec4(0.80f, 0.80f, 0.83f, 1.00f);
	colors[ImGuiCol_TextDisabled] = ImVec4(0.24f, 0.23f, 0.29f, 1.00f);
	colors[ImGuiCol_WindowBg] = ImVec4(0.06f, 0.05f, 0.07f, 1.00f);
	colors[ImGuiCol_PopupBg] = ImVec4(0.07f, 0.07f, 0.09f, 1.00f);
	colors[ImGuiCol_Border] = ImVec4(0.80f, 0.80f, 0.83f, 0.88f);
	colors[ImGuiCol_BorderShadow] = ImVec4(0.92f, 0.91f, 0.88f, 0.00f);
	colors[ImGuiCol_FrameBg] = ImVec4(0.10f, 0.09f, 0.12f, 1.00f);
	colors[ImGuiCol_FrameBgHovered] = ImVec4(0.24f, 0.23f, 0.29f, 1.00f);
	colors[ImGuiCol_FrameBgActive] = ImVec4(0.56f, 0.56f, 0.58f, 1.00f);
	colors[ImGuiCol_TitleBg] = ImVec4(0.10f, 0.09f, 0.12f, 1.00f);
	colors[ImGuiCol_TitleBgCollapsed] = ImVec4(1.00f, 0.98f, 0.95f, 0.75f);
	colors[ImGuiCol_TitleBgActive] = ImVec4(0.07f, 0.07f, 0.09f, 1.00f);
	colors[ImGuiCol_MenuBarBg] = ImVec4(0.10f, 0.09f, 0.12f, 1.00f);
	colors[ImGuiCol_ScrollbarBg] = ImVec4(0.10f, 0.09f, 0.12f, 1.00f);
	colors[ImGuiCol_ScrollbarGrab] = ImVec4(0.80f, 0.80f, 0.83f, 0.31f);
	colors[ImGuiCol_ScrollbarGrabHovered] = ImVec4(0.56f, 0.56f, 0.58f, 1.00f);
	colors[ImGuiCol_ScrollbarGrabActive] = ImVec4(0.06f, 0.05f, 0.07f, 1.00f);
	// colors[ImGuiCol_ComboBg] = ImVec4(0.19f, 0.18f, 0.21f, 1.00f);
	colors[ImGuiCol_CheckMark] = ImVec4(0.80f, 0.80f, 0.83f, 0.31f);
	colors[ImGuiCol_SliderGrab] = ImVec4(0.80f, 0.80f, 0.83f, 0.31f);
	colors[ImGuiCol_SliderGrabActive] = ImVec4(0.06f, 0.05f, 0.07f, 1.00f);
	colors[ImGuiCol_Button] = ImVec4(0.10f, 0.09f, 0.12f, 1.00f);
	colors[ImGuiCol_ButtonHovered] = ImVec4(0.24f, 0.23f, 0.29f, 1.00f);
	colors[ImGuiCol_ButtonActive] = ImVec4(0.56f, 0.56f, 0.58f, 1.00f);
	colors[ImGuiCol_Header] = ImVec4(0.10f, 0.09f, 0.12f, 1.00f);
	colors[ImGuiCol_HeaderHovered] = ImVec4(0.56f, 0.56f, 0.58f, 1.00f);
	colors[ImGuiCol_HeaderActive] = ImVec4(0.06f, 0.05f, 0.07f, 1.00f);
	// colors[ImGuiCol_Column] = ImVec4(0.56f, 0.56f, 0.58f, 1.00f);
	// colors[ImGuiCol_ColumnHovered] = ImVec4(0.24f, 0.23f, 0.29f, 1.00f);
	// colors[ImGuiCol_ColumnActive] = ImVec4(0.56f, 0.56f, 0.58f, 1.00f);
	colors[ImGuiCol_ResizeGrip] = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
	colors[ImGuiCol_ResizeGripHovered] = ImVec4(0.56f, 0.56f, 0.58f, 1.00f);
	colors[ImGuiCol_ResizeGripActive] = ImVec4(0.06f, 0.05f, 0.07f, 1.00f);
	// colors[ImGuiCol_CloseButton] = ImVec4(0.40f, 0.39f, 0.38f, 0.16f);
	// colors[ImGuiCol_CloseButtonHovered] = ImVec4(0.40f, 0.39f, 0.38f, 0.39f);
	// colors[ImGuiCol_CloseButtonActive] = ImVec4(0.40f, 0.39f, 0.38f, 1.00f);
	colors[ImGuiCol_PlotLines] = ImVec4(0.40f, 0.39f, 0.38f, 0.63f);
	colors[ImGuiCol_PlotLinesHovered] = ImVec4(0.25f, 1.00f, 0.00f, 1.00f);
	colors[ImGuiCol_PlotHistogram] = ImVec4(0.40f, 0.39f, 0.38f, 0.63f);
	colors[ImGuiCol_PlotHistogramHovered] = ImVec4(0.25f, 1.00f, 0.00f, 1.00f);
	colors[ImGuiCol_TextSelectedBg] = ImVec4(0.25f, 1.00f, 0.00f, 0.43f);
	//colors[ImGuiCol_ModalWindowDarkening] = ImVec4(1.00f, 0.98f, 0.95f, 0.73f);
}

void ImGui::StyleColorsGrey(ImGuiStyle* dst)
{
	ImGuiStyle* style = dst ? dst : &ImGui::GetStyle();
	ImVec4* colors = style->Colors;

	colors[ImGuiCol_Text] = ImVec4(1.00f, 1.00f, 1.00f, 1.00f);
	colors[ImGuiCol_TextDisabled] = ImVec4(0.50f, 0.50f, 0.50f, 1.00f);
	colors[ImGuiCol_WindowBg] = ImVec4(0.06f, 0.06f, 0.06f, 0.94f);
	colors[ImGuiCol_ChildBg] = ImVec4(1.00f, 1.00f, 1.00f, 0.00f);
	colors[ImGuiCol_PopupBg] = ImVec4(0.08f, 0.08f, 0.08f, 0.94f);
	colors[ImGuiCol_Border] = ImVec4(0.43f, 0.43f, 0.50f, 0.50f);
	colors[ImGuiCol_BorderShadow] = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
	colors[ImGuiCol_FrameBg] = ImVec4(0.20f, 0.21f, 0.22f, 0.54f);
	colors[ImGuiCol_FrameBgHovered] = ImVec4(0.40f, 0.40f, 0.40f, 0.40f);
	colors[ImGuiCol_FrameBgActive] = ImVec4(0.18f, 0.18f, 0.18f, 0.67f);
	colors[ImGuiCol_TitleBg] = ImVec4(0.04f, 0.04f, 0.04f, 1.00f);
	colors[ImGuiCol_TitleBgActive] = ImVec4(0.29f, 0.29f, 0.29f, 1.00f);
	colors[ImGuiCol_TitleBgCollapsed] = ImVec4(0.00f, 0.00f, 0.00f, 0.51f);
	colors[ImGuiCol_MenuBarBg] = ImVec4(0.14f, 0.14f, 0.14f, 1.00f);
	colors[ImGuiCol_ScrollbarBg] = ImVec4(0.02f, 0.02f, 0.02f, 0.53f);
	colors[ImGuiCol_ScrollbarGrab] = ImVec4(0.31f, 0.31f, 0.31f, 1.00f);
	colors[ImGuiCol_ScrollbarGrabHovered] = ImVec4(0.41f, 0.41f, 0.41f, 1.00f);
	colors[ImGuiCol_ScrollbarGrabActive] = ImVec4(0.51f, 0.51f, 0.51f, 1.00f);
	colors[ImGuiCol_CheckMark] = ImVec4(0.94f, 0.94f, 0.94f, 1.00f);
	colors[ImGuiCol_SliderGrab] = ImVec4(0.51f, 0.51f, 0.51f, 1.00f);
	colors[ImGuiCol_SliderGrabActive] = ImVec4(0.86f, 0.86f, 0.86f, 1.00f);
	colors[ImGuiCol_Button] = ImVec4(0.44f, 0.44f, 0.44f, 0.40f);
	colors[ImGuiCol_ButtonHovered] = ImVec4(0.46f, 0.47f, 0.48f, 1.00f);
	colors[ImGuiCol_ButtonActive] = ImVec4(0.42f, 0.42f, 0.42f, 1.00f);
	colors[ImGuiCol_Header] = ImVec4(0.70f, 0.70f, 0.70f, 0.31f);
	colors[ImGuiCol_HeaderHovered] = ImVec4(0.70f, 0.70f, 0.70f, 0.80f);
	colors[ImGuiCol_HeaderActive] = ImVec4(0.48f, 0.50f, 0.52f, 1.00f);
	colors[ImGuiCol_Separator] = ImVec4(0.43f, 0.43f, 0.50f, 0.50f);
	colors[ImGuiCol_SeparatorHovered] = ImVec4(0.72f, 0.72f, 0.72f, 0.78f);
	colors[ImGuiCol_SeparatorActive] = ImVec4(0.51f, 0.51f, 0.51f, 1.00f);
	colors[ImGuiCol_ResizeGrip] = ImVec4(0.91f, 0.91f, 0.91f, 0.25f);
	colors[ImGuiCol_ResizeGripHovered] = ImVec4(0.81f, 0.81f, 0.81f, 0.67f);
	colors[ImGuiCol_ResizeGripActive] = ImVec4(0.46f, 0.46f, 0.46f, 0.95f);
	colors[ImGuiCol_PlotLines] = ImVec4(0.61f, 0.61f, 0.61f, 1.00f);
	colors[ImGuiCol_PlotLinesHovered] = ImVec4(1.00f, 0.43f, 0.35f, 1.00f);
	colors[ImGuiCol_PlotHistogram] = ImVec4(0.73f, 0.60f, 0.15f, 1.00f);
	colors[ImGuiCol_PlotHistogramHovered] = ImVec4(1.00f, 0.60f, 0.00f, 1.00f);
	colors[ImGuiCol_TextSelectedBg] = ImVec4(0.87f, 0.87f, 0.87f, 0.35f);
	//colors[ImGuiCol_ModalWindowDarkening] = ImVec4(0.80f, 0.80f, 0.80f, 0.35f);
	colors[ImGuiCol_DragDropTarget] = ImVec4(1.00f, 1.00f, 0.00f, 0.90f);
	colors[ImGuiCol_NavHighlight] = ImVec4(0.60f, 0.60f, 0.60f, 1.00f);
	colors[ImGuiCol_NavWindowingHighlight] = ImVec4(1.00f, 1.00f, 1.00f, 0.70f);
}

void ImGui::StyleColorsCharcoal(ImGuiStyle* dst)
{
	ImGuiStyle* style = dst ? dst : &ImGui::GetStyle();
	ImVec4* colors = style->Colors;

	colors[ImGuiCol_Text] = ImVec4(1.000f, 1.000f, 1.000f, 1.000f);
	colors[ImGuiCol_TextDisabled] = ImVec4(0.500f, 0.500f, 0.500f, 1.000f);
	colors[ImGuiCol_WindowBg] = ImVec4(0.180f, 0.180f, 0.180f, 1.000f);
	colors[ImGuiCol_ChildBg] = ImVec4(0.280f, 0.280f, 0.280f, 0.000f);
	colors[ImGuiCol_PopupBg] = ImVec4(0.313f, 0.313f, 0.313f, 1.000f);
	colors[ImGuiCol_Border] = ImVec4(0.266f, 0.266f, 0.266f, 1.000f);
	colors[ImGuiCol_BorderShadow] = ImVec4(0.000f, 0.000f, 0.000f, 0.000f);
	colors[ImGuiCol_FrameBg] = ImVec4(0.160f, 0.160f, 0.160f, 1.000f);
	colors[ImGuiCol_FrameBgHovered] = ImVec4(0.200f, 0.200f, 0.200f, 1.000f);
	colors[ImGuiCol_FrameBgActive] = ImVec4(0.280f, 0.280f, 0.280f, 1.000f);
	colors[ImGuiCol_TitleBg] = ImVec4(0.148f, 0.148f, 0.148f, 1.000f);
	colors[ImGuiCol_TitleBgActive] = ImVec4(0.148f, 0.148f, 0.148f, 1.000f);
	colors[ImGuiCol_TitleBgCollapsed] = ImVec4(0.148f, 0.148f, 0.148f, 1.000f);
	colors[ImGuiCol_MenuBarBg] = ImVec4(0.195f, 0.195f, 0.195f, 1.000f);
	colors[ImGuiCol_ScrollbarBg] = ImVec4(0.160f, 0.160f, 0.160f, 1.000f);
	colors[ImGuiCol_ScrollbarGrab] = ImVec4(0.277f, 0.277f, 0.277f, 1.000f);
	colors[ImGuiCol_ScrollbarGrabHovered] = ImVec4(0.300f, 0.300f, 0.300f, 1.000f);
	colors[ImGuiCol_ScrollbarGrabActive] = ImVec4(1.000f, 0.391f, 0.000f, 1.000f);
	colors[ImGuiCol_CheckMark] = ImVec4(1.000f, 1.000f, 1.000f, 1.000f);
	colors[ImGuiCol_SliderGrab] = ImVec4(0.391f, 0.391f, 0.391f, 1.000f);
	colors[ImGuiCol_SliderGrabActive] = ImVec4(1.000f, 0.391f, 0.000f, 1.000f);
	colors[ImGuiCol_Button] = ImVec4(1.000f, 1.000f, 1.000f, 0.000f);
	colors[ImGuiCol_ButtonHovered] = ImVec4(1.000f, 1.000f, 1.000f, 0.156f);
	colors[ImGuiCol_ButtonActive] = ImVec4(1.000f, 1.000f, 1.000f, 0.391f);
	colors[ImGuiCol_Header] = ImVec4(0.313f, 0.313f, 0.313f, 1.000f);
	colors[ImGuiCol_HeaderHovered] = ImVec4(0.469f, 0.469f, 0.469f, 1.000f);
	colors[ImGuiCol_HeaderActive] = ImVec4(0.469f, 0.469f, 0.469f, 1.000f);
	colors[ImGuiCol_Separator] = colors[ImGuiCol_Border];
	colors[ImGuiCol_SeparatorHovered] = ImVec4(0.391f, 0.391f, 0.391f, 1.000f);
	colors[ImGuiCol_SeparatorActive] = ImVec4(1.000f, 0.391f, 0.000f, 1.000f);
	colors[ImGuiCol_ResizeGrip] = ImVec4(1.000f, 1.000f, 1.000f, 0.250f);
	colors[ImGuiCol_ResizeGripHovered] = ImVec4(1.000f, 1.000f, 1.000f, 0.670f);
	colors[ImGuiCol_ResizeGripActive] = ImVec4(1.000f, 0.391f, 0.000f, 1.000f);
	colors[ImGuiCol_Tab] = ImVec4(0.098f, 0.098f, 0.098f, 1.000f);
	colors[ImGuiCol_TabHovered] = ImVec4(0.352f, 0.352f, 0.352f, 1.000f);
	colors[ImGuiCol_TabActive] = ImVec4(0.195f, 0.195f, 0.195f, 1.000f);
	colors[ImGuiCol_TabUnfocused] = ImVec4(0.098f, 0.098f, 0.098f, 1.000f);
	colors[ImGuiCol_TabUnfocusedActive] = ImVec4(0.195f, 0.195f, 0.195f, 1.000f);
	colors[ImGuiCol_DockingPreview] = ImVec4(1.000f, 0.391f, 0.000f, 0.781f);
	colors[ImGuiCol_DockingEmptyBg] = ImVec4(0.180f, 0.180f, 0.180f, 1.000f);
	colors[ImGuiCol_PlotLines] = ImVec4(0.469f, 0.469f, 0.469f, 1.000f);
	colors[ImGuiCol_PlotLinesHovered] = ImVec4(1.000f, 0.391f, 0.000f, 1.000f);
	colors[ImGuiCol_PlotHistogram] = ImVec4(0.586f, 0.586f, 0.586f, 1.000f);
	colors[ImGuiCol_PlotHistogramHovered] = ImVec4(1.000f, 0.391f, 0.000f, 1.000f);
	colors[ImGuiCol_TextSelectedBg] = ImVec4(1.000f, 1.000f, 1.000f, 0.156f);
	colors[ImGuiCol_DragDropTarget] = ImVec4(1.000f, 0.391f, 0.000f, 1.000f);
	colors[ImGuiCol_NavHighlight] = ImVec4(1.000f, 0.391f, 0.000f, 1.000f);
	colors[ImGuiCol_NavWindowingHighlight] = ImVec4(1.000f, 0.391f, 0.000f, 1.000f);
	colors[ImGuiCol_NavWindowingDimBg] = ImVec4(0.000f, 0.000f, 0.000f, 0.586f);
	colors[ImGuiCol_ModalWindowDimBg] = ImVec4(0.000f, 0.000f, 0.000f, 0.586f);

	style->ChildRounding = 4.0f;
	style->FrameBorderSize = 1.0f;
	style->FrameRounding = 2.0f;
	style->GrabMinSize = 7.0f;
	style->PopupRounding = 2.0f;
	style->ScrollbarRounding = 12.0f;
	style->ScrollbarSize = 13.0f;
	style->TabBorderSize = 1.0f;
	style->TabRounding = 0.0f;
	style->WindowRounding = 4.0f;
}

void ImGui::StyleColorsCorporate(ImGuiStyle* dst)
{
	ImGuiStyle* style = dst ? dst : &ImGui::GetStyle();
	ImVec4* colors = style->Colors;

	/// 0 = FLAT APPEARENCE
	/// 1 = MORE "3D" LOOK
	float is3D = 0.0f;

	colors[ImGuiCol_Text] = ImVec4(1.00f, 1.00f, 1.00f, 1.00f);
	colors[ImGuiCol_TextDisabled] = ImVec4(0.40f, 0.40f, 0.40f, 1.00f);
	colors[ImGuiCol_ChildBg] = ImVec4(0.25f, 0.25f, 0.25f, 1.00f);
	colors[ImGuiCol_WindowBg] = ImVec4(0.25f, 0.25f, 0.25f, 1.00f);
	colors[ImGuiCol_PopupBg] = ImVec4(0.25f, 0.25f, 0.25f, 1.00f);
	colors[ImGuiCol_Border] = ImVec4(0.12f, 0.12f, 0.12f, 0.71f);
	colors[ImGuiCol_BorderShadow] = ImVec4(1.00f, 1.00f, 1.00f, 0.06f);
	colors[ImGuiCol_FrameBg] = ImVec4(0.42f, 0.42f, 0.42f, 0.54f);
	colors[ImGuiCol_FrameBgHovered] = ImVec4(0.42f, 0.42f, 0.42f, 0.40f);
	colors[ImGuiCol_FrameBgActive] = ImVec4(0.56f, 0.56f, 0.56f, 0.67f);
	colors[ImGuiCol_TitleBg] = ImVec4(0.19f, 0.19f, 0.19f, 1.00f);
	colors[ImGuiCol_TitleBgActive] = ImVec4(0.22f, 0.22f, 0.22f, 1.00f);
	colors[ImGuiCol_TitleBgCollapsed] = ImVec4(0.17f, 0.17f, 0.17f, 0.90f);
	colors[ImGuiCol_MenuBarBg] = ImVec4(0.335f, 0.335f, 0.335f, 1.000f);
	colors[ImGuiCol_ScrollbarBg] = ImVec4(0.24f, 0.24f, 0.24f, 0.53f);
	colors[ImGuiCol_ScrollbarGrab] = ImVec4(0.41f, 0.41f, 0.41f, 1.00f);
	colors[ImGuiCol_ScrollbarGrabHovered] = ImVec4(0.52f, 0.52f, 0.52f, 1.00f);
	colors[ImGuiCol_ScrollbarGrabActive] = ImVec4(0.76f, 0.76f, 0.76f, 1.00f);
	colors[ImGuiCol_CheckMark] = ImVec4(0.65f, 0.65f, 0.65f, 1.00f);
	colors[ImGuiCol_SliderGrab] = ImVec4(0.52f, 0.52f, 0.52f, 1.00f);
	colors[ImGuiCol_SliderGrabActive] = ImVec4(0.64f, 0.64f, 0.64f, 1.00f);
	colors[ImGuiCol_Button] = ImVec4(0.54f, 0.54f, 0.54f, 0.35f);
	colors[ImGuiCol_ButtonHovered] = ImVec4(0.52f, 0.52f, 0.52f, 0.59f);
	colors[ImGuiCol_ButtonActive] = ImVec4(0.76f, 0.76f, 0.76f, 1.00f);
	colors[ImGuiCol_Header] = ImVec4(0.38f, 0.38f, 0.38f, 1.00f);
	colors[ImGuiCol_HeaderHovered] = ImVec4(0.47f, 0.47f, 0.47f, 1.00f);
	colors[ImGuiCol_HeaderActive] = ImVec4(0.76f, 0.76f, 0.76f, 0.77f);
	colors[ImGuiCol_Separator] = ImVec4(0.000f, 0.000f, 0.000f, 0.137f);
	colors[ImGuiCol_SeparatorHovered] = ImVec4(0.700f, 0.671f, 0.600f, 0.290f);
	colors[ImGuiCol_SeparatorActive] = ImVec4(0.702f, 0.671f, 0.600f, 0.674f);
	colors[ImGuiCol_ResizeGrip] = ImVec4(0.26f, 0.59f, 0.98f, 0.25f);
	colors[ImGuiCol_ResizeGripHovered] = ImVec4(0.26f, 0.59f, 0.98f, 0.67f);
	colors[ImGuiCol_ResizeGripActive] = ImVec4(0.26f, 0.59f, 0.98f, 0.95f);
	colors[ImGuiCol_PlotLines] = ImVec4(0.61f, 0.61f, 0.61f, 1.00f);
	colors[ImGuiCol_PlotLinesHovered] = ImVec4(1.00f, 0.43f, 0.35f, 1.00f);
	colors[ImGuiCol_PlotHistogram] = ImVec4(0.90f, 0.70f, 0.00f, 1.00f);
	colors[ImGuiCol_PlotHistogramHovered] = ImVec4(1.00f, 0.60f, 0.00f, 1.00f);
	colors[ImGuiCol_TextSelectedBg] = ImVec4(0.73f, 0.73f, 0.73f, 0.35f);
	colors[ImGuiCol_ModalWindowDimBg] = ImVec4(0.80f, 0.80f, 0.80f, 0.35f);
	colors[ImGuiCol_DragDropTarget] = ImVec4(1.00f, 1.00f, 0.00f, 0.90f);
	colors[ImGuiCol_NavHighlight] = ImVec4(0.26f, 0.59f, 0.98f, 1.00f);
	colors[ImGuiCol_NavWindowingHighlight] = ImVec4(1.00f, 1.00f, 1.00f, 0.70f);
	colors[ImGuiCol_NavWindowingDimBg] = ImVec4(0.80f, 0.80f, 0.80f, 0.20f);

	style->PopupRounding = 3;

	style->WindowPadding = ImVec2(4, 4);
	style->FramePadding = ImVec2(6, 4);
	style->ItemSpacing = ImVec2(6, 2);

	style->ScrollbarSize = 18;

	style->WindowBorderSize = 1;
	style->ChildBorderSize = 1;
	style->PopupBorderSize = 1;
	style->FrameBorderSize = is3D;

	style->WindowRounding = 3;
	style->ChildRounding = 3;
	style->FrameRounding = 3;
	style->ScrollbarRounding = 2;
	style->GrabRounding = 3;

#ifdef IMGUI_HAS_DOCK
	style->TabBorderSize = is3D;
	style->TabRounding = 3;

	colors[ImGuiCol_DockingEmptyBg] = ImVec4(0.38f, 0.38f, 0.38f, 1.00f);
	colors[ImGuiCol_Tab] = ImVec4(0.25f, 0.25f, 0.25f, 1.00f);
	colors[ImGuiCol_TabHovered] = ImVec4(0.40f, 0.40f, 0.40f, 1.00f);
	colors[ImGuiCol_TabActive] = ImVec4(0.33f, 0.33f, 0.33f, 1.00f);
	colors[ImGuiCol_TabUnfocused] = ImVec4(0.25f, 0.25f, 0.25f, 1.00f);
	colors[ImGuiCol_TabUnfocusedActive] = ImVec4(0.33f, 0.33f, 0.33f, 1.00f);
	colors[ImGuiCol_DockingPreview] = ImVec4(0.85f, 0.85f, 0.85f, 0.28f);

	if (ImGui::GetIO().ConfigFlags & ImGuiConfigFlags_ViewportsEnable)
	{
		style->WindowRounding = 0.0f;
		style->Colors[ImGuiCol_WindowBg].w = 1.0f;
	}
#endif
}


void ImGui::StyleColorsCinder(ImGuiStyle* dst)
{
	ImGuiStyle* style = dst ? dst : &ImGui::GetStyle();
	ImVec4* colors = style->Colors;

	style->WindowMinSize = ImVec2(160, 20);
	style->FramePadding = ImVec2(4, 2);
	style->ItemSpacing = ImVec2(6, 2);
	style->ItemInnerSpacing = ImVec2(6, 4);
	style->Alpha = 0.95f;
	style->WindowRounding = 4.0f;
	style->FrameRounding = 2.0f;
	style->IndentSpacing = 6.0f;
	style->ItemInnerSpacing = ImVec2(2, 4);
	style->ColumnsMinSpacing = 50.0f;
	style->GrabMinSize = 14.0f;
	style->GrabRounding = 16.0f;
	style->ScrollbarSize = 12.0f;
	style->ScrollbarRounding = 16.0f;

	colors[ImGuiCol_Text] = ImVec4(0.86f, 0.93f, 0.89f, 0.78f);
	colors[ImGuiCol_TextDisabled] = ImVec4(0.86f, 0.93f, 0.89f, 0.28f);
	colors[ImGuiCol_WindowBg] = ImVec4(0.13f, 0.14f, 0.17f, 1.00f);
	colors[ImGuiCol_Border] = ImVec4(0.31f, 0.31f, 1.00f, 0.00f);
	colors[ImGuiCol_BorderShadow] = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
	colors[ImGuiCol_FrameBg] = ImVec4(0.20f, 0.22f, 0.27f, 1.00f);
	colors[ImGuiCol_FrameBgHovered] = ImVec4(0.92f, 0.18f, 0.29f, 0.78f);
	colors[ImGuiCol_FrameBgActive] = ImVec4(0.92f, 0.18f, 0.29f, 1.00f);
	colors[ImGuiCol_TitleBg] = ImVec4(0.20f, 0.22f, 0.27f, 1.00f);
	colors[ImGuiCol_TitleBgCollapsed] = ImVec4(0.20f, 0.22f, 0.27f, 0.75f);
	colors[ImGuiCol_TitleBgActive] = ImVec4(0.92f, 0.18f, 0.29f, 1.00f);
	colors[ImGuiCol_MenuBarBg] = ImVec4(0.20f, 0.22f, 0.27f, 0.47f);
	colors[ImGuiCol_ScrollbarBg] = ImVec4(0.20f, 0.22f, 0.27f, 1.00f);
	colors[ImGuiCol_ScrollbarGrab] = ImVec4(0.09f, 0.15f, 0.16f, 1.00f);
	colors[ImGuiCol_ScrollbarGrabHovered] = ImVec4(0.92f, 0.18f, 0.29f, 0.78f);
	colors[ImGuiCol_ScrollbarGrabActive] = ImVec4(0.92f, 0.18f, 0.29f, 1.00f);
	colors[ImGuiCol_CheckMark] = ImVec4(0.71f, 0.22f, 0.27f, 1.00f);
	colors[ImGuiCol_SliderGrab] = ImVec4(0.47f, 0.77f, 0.83f, 0.14f);
	colors[ImGuiCol_SliderGrabActive] = ImVec4(0.92f, 0.18f, 0.29f, 1.00f);
	colors[ImGuiCol_Button] = ImVec4(0.47f, 0.77f, 0.83f, 0.14f);
	colors[ImGuiCol_ButtonHovered] = ImVec4(0.92f, 0.18f, 0.29f, 0.86f);
	colors[ImGuiCol_ButtonActive] = ImVec4(0.92f, 0.18f, 0.29f, 1.00f);
	colors[ImGuiCol_Header] = ImVec4(0.92f, 0.18f, 0.29f, 0.76f);
	colors[ImGuiCol_HeaderHovered] = ImVec4(0.92f, 0.18f, 0.29f, 0.86f);
	colors[ImGuiCol_HeaderActive] = ImVec4(0.92f, 0.18f, 0.29f, 1.00f);
	colors[ImGuiCol_Separator] = ImVec4(0.14f, 0.16f, 0.19f, 1.00f);
	colors[ImGuiCol_SeparatorHovered] = ImVec4(0.92f, 0.18f, 0.29f, 0.78f);
	colors[ImGuiCol_SeparatorActive] = ImVec4(0.92f, 0.18f, 0.29f, 1.00f);
	colors[ImGuiCol_ResizeGrip] = ImVec4(0.47f, 0.77f, 0.83f, 0.04f);
	colors[ImGuiCol_ResizeGripHovered] = ImVec4(0.92f, 0.18f, 0.29f, 0.78f);
	colors[ImGuiCol_ResizeGripActive] = ImVec4(0.92f, 0.18f, 0.29f, 1.00f);
	colors[ImGuiCol_PlotLines] = ImVec4(0.86f, 0.93f, 0.89f, 0.63f);
	colors[ImGuiCol_PlotLinesHovered] = ImVec4(0.92f, 0.18f, 0.29f, 1.00f);
	colors[ImGuiCol_PlotHistogram] = ImVec4(0.86f, 0.93f, 0.89f, 0.63f);
	colors[ImGuiCol_PlotHistogramHovered] = ImVec4(0.92f, 0.18f, 0.29f, 1.00f);
	colors[ImGuiCol_TextSelectedBg] = ImVec4(0.92f, 0.18f, 0.29f, 0.43f);
	colors[ImGuiCol_PopupBg] = ImVec4(0.20f, 0.22f, 0.27f, 0.9f);
	//colors[ImGuiCol_ModalWindowDarkening] = ImVec4(0.20f, 0.22f, 0.27f, 0.73f);
}

void ImGui::StyleColorsEnemyMouse(ImGuiStyle* dst)
{
	ImGuiStyle* style = dst ? dst : &ImGui::GetStyle();
	ImVec4* colors = style->Colors;

	style->Alpha = 1.0;
	//style->WindowFillAlphaDefault = 0.83;
	//style->ChildWindowRounding = 3;
	style->WindowRounding = 3;
	style->GrabRounding = 1;
	style->GrabMinSize = 20;
	style->FrameRounding = 3;


	colors[ImGuiCol_Text] = ImVec4(0.00f, 1.00f, 1.00f, 1.00f);
	colors[ImGuiCol_TextDisabled] = ImVec4(0.00f, 0.40f, 0.41f, 1.00f);
	colors[ImGuiCol_WindowBg] = ImVec4(0.00f, 0.00f, 0.00f, 1.00f);
	//colors[ImGuiCol_ChildWindowBg] = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
	colors[ImGuiCol_Border] = ImVec4(0.00f, 1.00f, 1.00f, 0.65f);
	colors[ImGuiCol_BorderShadow] = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
	colors[ImGuiCol_FrameBg] = ImVec4(0.44f, 0.80f, 0.80f, 0.18f);
	colors[ImGuiCol_FrameBgHovered] = ImVec4(0.44f, 0.80f, 0.80f, 0.27f);
	colors[ImGuiCol_FrameBgActive] = ImVec4(0.44f, 0.81f, 0.86f, 0.66f);
	colors[ImGuiCol_TitleBg] = ImVec4(0.14f, 0.18f, 0.21f, 0.73f);
	colors[ImGuiCol_TitleBgCollapsed] = ImVec4(0.00f, 0.00f, 0.00f, 0.54f);
	colors[ImGuiCol_TitleBgActive] = ImVec4(0.00f, 1.00f, 1.00f, 0.27f);
	colors[ImGuiCol_MenuBarBg] = ImVec4(0.00f, 0.00f, 0.00f, 0.20f);
	colors[ImGuiCol_ScrollbarBg] = ImVec4(0.22f, 0.29f, 0.30f, 0.71f);
	colors[ImGuiCol_ScrollbarGrab] = ImVec4(0.00f, 1.00f, 1.00f, 0.44f);
	colors[ImGuiCol_ScrollbarGrabHovered] = ImVec4(0.00f, 1.00f, 1.00f, 0.74f);
	colors[ImGuiCol_ScrollbarGrabActive] = ImVec4(0.00f, 1.00f, 1.00f, 1.00f);
	//colors[ImGuiCol_ComboBg] = ImVec4(0.16f, 0.24f, 0.22f, 0.60f);
	colors[ImGuiCol_CheckMark] = ImVec4(0.00f, 1.00f, 1.00f, 0.68f);
	colors[ImGuiCol_SliderGrab] = ImVec4(0.00f, 1.00f, 1.00f, 0.36f);
	colors[ImGuiCol_SliderGrabActive] = ImVec4(0.00f, 1.00f, 1.00f, 0.76f);
	colors[ImGuiCol_Button] = ImVec4(0.00f, 0.65f, 0.65f, 0.46f);
	colors[ImGuiCol_ButtonHovered] = ImVec4(0.01f, 1.00f, 1.00f, 0.43f);
	colors[ImGuiCol_ButtonActive] = ImVec4(0.00f, 1.00f, 1.00f, 0.62f);
	colors[ImGuiCol_Header] = ImVec4(0.00f, 1.00f, 1.00f, 0.33f);
	colors[ImGuiCol_HeaderHovered] = ImVec4(0.00f, 1.00f, 1.00f, 0.42f);
	colors[ImGuiCol_HeaderActive] = ImVec4(0.00f, 1.00f, 1.00f, 0.54f);
	//colors[ImGuiCol_Column] = ImVec4(0.00f, 0.50f, 0.50f, 0.33f);
	//colors[ImGuiCol_ColumnHovered] = ImVec4(0.00f, 0.50f, 0.50f, 0.47f);
	//colors[ImGuiCol_ColumnActive] = ImVec4(0.00f, 0.70f, 0.70f, 1.00f);
	colors[ImGuiCol_ResizeGrip] = ImVec4(0.00f, 1.00f, 1.00f, 0.54f);
	colors[ImGuiCol_ResizeGripHovered] = ImVec4(0.00f, 1.00f, 1.00f, 0.74f);
	colors[ImGuiCol_ResizeGripActive] = ImVec4(0.00f, 1.00f, 1.00f, 1.00f);
	//colors[ImGuiCol_CloseButton] = ImVec4(0.00f, 0.78f, 0.78f, 0.35f);
	//colors[ImGuiCol_CloseButtonHovered] = ImVec4(0.00f, 0.78f, 0.78f, 0.47f);
	//colors[ImGuiCol_CloseButtonActive] = ImVec4(0.00f, 0.78f, 0.78f, 1.00f);
	colors[ImGuiCol_PlotLines] = ImVec4(0.00f, 1.00f, 1.00f, 1.00f);
	colors[ImGuiCol_PlotLinesHovered] = ImVec4(0.00f, 1.00f, 1.00f, 1.00f);
	colors[ImGuiCol_PlotHistogram] = ImVec4(0.00f, 1.00f, 1.00f, 1.00f);
	colors[ImGuiCol_PlotHistogramHovered] = ImVec4(0.00f, 1.00f, 1.00f, 1.00f);
	colors[ImGuiCol_TextSelectedBg] = ImVec4(0.00f, 1.00f, 1.00f, 0.22f);
	//colors[ImGuiCol_TooltipBg] = ImVec4(0.00f, 0.13f, 0.13f, 0.90f);
	//colors[ImGuiCol_ModalWindowDarkening] = ImVec4(0.04f, 0.10f, 0.09f, 0.51f);
}

void ImGui::StyleColorsDougBlinks(ImGuiStyle* dst)
{
	ImGuiStyle* style = dst ? dst : &ImGui::GetStyle();
	ImVec4* colors = style->Colors;

	// light style from Pacôme Danhiez (user itamago) https://github.com/ocornut/imgui/pull/511#issuecomment-175719267
	style->Alpha = 1.0f;
	style->FrameRounding = 3.0f;

	colors[ImGuiCol_Text] = ImVec4(0.00f, 0.00f, 0.00f, 1.00f);
	colors[ImGuiCol_TextDisabled] = ImVec4(0.60f, 0.60f, 0.60f, 1.00f);
	colors[ImGuiCol_WindowBg] = ImVec4(0.94f, 0.94f, 0.94f, 0.94f);
	//colors[ImGuiCol_ChildWindowBg] = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
	colors[ImGuiCol_PopupBg] = ImVec4(1.00f, 1.00f, 1.00f, 0.94f);
	colors[ImGuiCol_Border] = ImVec4(0.00f, 0.00f, 0.00f, 0.39f);
	colors[ImGuiCol_BorderShadow] = ImVec4(1.00f, 1.00f, 1.00f, 0.10f);
	colors[ImGuiCol_FrameBg] = ImVec4(1.00f, 1.00f, 1.00f, 0.94f);
	colors[ImGuiCol_FrameBgHovered] = ImVec4(0.26f, 0.59f, 0.98f, 0.40f);
	colors[ImGuiCol_FrameBgActive] = ImVec4(0.26f, 0.59f, 0.98f, 0.67f);
	colors[ImGuiCol_TitleBg] = ImVec4(0.96f, 0.96f, 0.96f, 1.00f);
	colors[ImGuiCol_TitleBgCollapsed] = ImVec4(1.00f, 1.00f, 1.00f, 0.51f);
	colors[ImGuiCol_TitleBgActive] = ImVec4(0.82f, 0.82f, 0.82f, 1.00f);
	colors[ImGuiCol_MenuBarBg] = ImVec4(0.86f, 0.86f, 0.86f, 1.00f);
	colors[ImGuiCol_ScrollbarBg] = ImVec4(0.98f, 0.98f, 0.98f, 0.53f);
	colors[ImGuiCol_ScrollbarGrab] = ImVec4(0.69f, 0.69f, 0.69f, 1.00f);
	colors[ImGuiCol_ScrollbarGrabHovered] = ImVec4(0.59f, 0.59f, 0.59f, 1.00f);
	colors[ImGuiCol_ScrollbarGrabActive] = ImVec4(0.49f, 0.49f, 0.49f, 1.00f);
	//colors[ImGuiCol_ComboBg] = ImVec4(0.86f, 0.86f, 0.86f, 0.99f);
	colors[ImGuiCol_CheckMark] = ImVec4(0.26f, 0.59f, 0.98f, 1.00f);
	colors[ImGuiCol_SliderGrab] = ImVec4(0.24f, 0.52f, 0.88f, 1.00f);
	colors[ImGuiCol_SliderGrabActive] = ImVec4(0.26f, 0.59f, 0.98f, 1.00f);
	colors[ImGuiCol_Button] = ImVec4(0.26f, 0.59f, 0.98f, 0.40f);
	colors[ImGuiCol_ButtonHovered] = ImVec4(0.26f, 0.59f, 0.98f, 1.00f);
	colors[ImGuiCol_ButtonActive] = ImVec4(0.06f, 0.53f, 0.98f, 1.00f);
	colors[ImGuiCol_Header] = ImVec4(0.26f, 0.59f, 0.98f, 0.31f);
	colors[ImGuiCol_HeaderHovered] = ImVec4(0.26f, 0.59f, 0.98f, 0.80f);
	colors[ImGuiCol_HeaderActive] = ImVec4(0.26f, 0.59f, 0.98f, 1.00f);
	//colors[ImGuiCol_Column] = ImVec4(0.39f, 0.39f, 0.39f, 1.00f);
	//colors[ImGuiCol_ColumnHovered] = ImVec4(0.26f, 0.59f, 0.98f, 0.78f);
	//colors[ImGuiCol_ColumnActive] = ImVec4(0.26f, 0.59f, 0.98f, 1.00f);
	colors[ImGuiCol_ResizeGrip] = ImVec4(1.00f, 1.00f, 1.00f, 0.50f);
	colors[ImGuiCol_ResizeGripHovered] = ImVec4(0.26f, 0.59f, 0.98f, 0.67f);
	colors[ImGuiCol_ResizeGripActive] = ImVec4(0.26f, 0.59f, 0.98f, 0.95f);
	//colors[ImGuiCol_CloseButton] = ImVec4(0.59f, 0.59f, 0.59f, 0.50f);
	//colors[ImGuiCol_CloseButtonHovered] = ImVec4(0.98f, 0.39f, 0.36f, 1.00f);
	//colors[ImGuiCol_CloseButtonActive] = ImVec4(0.98f, 0.39f, 0.36f, 1.00f);
	colors[ImGuiCol_PlotLines] = ImVec4(0.39f, 0.39f, 0.39f, 1.00f);
	colors[ImGuiCol_PlotLinesHovered] = ImVec4(1.00f, 0.43f, 0.35f, 1.00f);
	colors[ImGuiCol_PlotHistogram] = ImVec4(0.90f, 0.70f, 0.00f, 1.00f);
	colors[ImGuiCol_PlotHistogramHovered] = ImVec4(1.00f, 0.60f, 0.00f, 1.00f);
	colors[ImGuiCol_TextSelectedBg] = ImVec4(0.26f, 0.59f, 0.98f, 0.35f);
	//colors[ImGuiCol_ModalWindowDarkening] = ImVec4(0.20f, 0.20f, 0.20f, 0.35f);

	/*if (bStyleDark_)
	{
		for (int i = 0; i <= ImGuiCol_COUNT; i++)
		{
			ImVec4& col = colors[i];
			float H, S, V;
			ImGui::ColorConvertRGBtoHSV(col.x, col.y, col.z, H, S, V);

			if (S < 0.1f)
			{
				V = 1.0f - V;
			}
			ImGui::ColorConvertHSVtoRGB(H, S, V, col.x, col.y, col.z);
			if (col.w < 1.00f)
			{
				col.w *= alpha_;
			}
		}
	}
	else
	{
		for (int i = 0; i <= ImGuiCol_COUNT; i++)
		{
			ImVec4& col = colors[i];
			if (col.w < 1.00f)
			{
				col.x *= alpha_;
				col.y *= alpha_;
				col.z *= alpha_;
				col.w *= alpha_;
			}
		}
	}*/
}

void ImGui::StyleColorsGreenBlue(ImGuiStyle* dst)
{
	ImGuiStyle* style = dst ? dst : &ImGui::GetStyle();
	ImVec4* colors = style->Colors;

	colors[ImGuiCol_Text] = ImVec4(1.00f, 1.00f, 1.00f, 1.00f);
	colors[ImGuiCol_TextDisabled] = ImVec4(0.50f, 0.50f, 0.50f, 1.00f);
	colors[ImGuiCol_WindowBg] = ImVec4(0.06f, 0.06f, 0.06f, 0.94f);
	colors[ImGuiCol_ChildBg] = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
	colors[ImGuiCol_PopupBg] = ImVec4(0.08f, 0.08f, 0.08f, 0.94f);
	colors[ImGuiCol_Border] = ImVec4(0.43f, 0.43f, 0.50f, 0.50f);
	colors[ImGuiCol_BorderShadow] = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
	colors[ImGuiCol_FrameBg] = ImVec4(0.44f, 0.44f, 0.44f, 0.60f);
	colors[ImGuiCol_FrameBgHovered] = ImVec4(0.57f, 0.57f, 0.57f, 0.70f);
	colors[ImGuiCol_FrameBgActive] = ImVec4(0.76f, 0.76f, 0.76f, 0.80f);
	colors[ImGuiCol_TitleBg] = ImVec4(0.04f, 0.04f, 0.04f, 1.00f);
	colors[ImGuiCol_TitleBgActive] = ImVec4(0.16f, 0.16f, 0.16f, 1.00f);
	colors[ImGuiCol_TitleBgCollapsed] = ImVec4(0.00f, 0.00f, 0.00f, 0.60f);
	colors[ImGuiCol_MenuBarBg] = ImVec4(0.14f, 0.14f, 0.14f, 1.00f);
	colors[ImGuiCol_ScrollbarBg] = ImVec4(0.02f, 0.02f, 0.02f, 0.53f);
	colors[ImGuiCol_ScrollbarGrab] = ImVec4(0.31f, 0.31f, 0.31f, 1.00f);
	colors[ImGuiCol_ScrollbarGrabHovered] = ImVec4(0.41f, 0.41f, 0.41f, 1.00f);
	colors[ImGuiCol_ScrollbarGrabActive] = ImVec4(0.51f, 0.51f, 0.51f, 1.00f);
	colors[ImGuiCol_CheckMark] = ImVec4(0.13f, 0.75f, 0.55f, 0.80f);
	colors[ImGuiCol_SliderGrab] = ImVec4(0.13f, 0.75f, 0.75f, 0.80f);
	colors[ImGuiCol_SliderGrabActive] = ImVec4(0.13f, 0.75f, 1.00f, 0.80f);
	colors[ImGuiCol_Button] = ImVec4(0.13f, 0.75f, 0.55f, 0.40f);
	colors[ImGuiCol_ButtonHovered] = ImVec4(0.13f, 0.75f, 0.75f, 0.60f);
	colors[ImGuiCol_ButtonActive] = ImVec4(0.13f, 0.75f, 1.00f, 0.80f);
	colors[ImGuiCol_Header] = ImVec4(0.13f, 0.75f, 0.55f, 0.40f);
	colors[ImGuiCol_HeaderHovered] = ImVec4(0.13f, 0.75f, 0.75f, 0.60f);
	colors[ImGuiCol_HeaderActive] = ImVec4(0.13f, 0.75f, 1.00f, 0.80f);
	colors[ImGuiCol_Separator] = ImVec4(0.13f, 0.75f, 0.55f, 0.40f);
	colors[ImGuiCol_SeparatorHovered] = ImVec4(0.13f, 0.75f, 0.75f, 0.60f);
	colors[ImGuiCol_SeparatorActive] = ImVec4(0.13f, 0.75f, 1.00f, 0.80f);
	colors[ImGuiCol_ResizeGrip] = ImVec4(0.13f, 0.75f, 0.55f, 0.40f);
	colors[ImGuiCol_ResizeGripHovered] = ImVec4(0.13f, 0.75f, 0.75f, 0.60f);
	colors[ImGuiCol_ResizeGripActive] = ImVec4(0.13f, 0.75f, 1.00f, 0.80f);
	colors[ImGuiCol_Tab] = ImVec4(0.13f, 0.75f, 0.55f, 0.80f);
	colors[ImGuiCol_TabHovered] = ImVec4(0.13f, 0.75f, 0.75f, 0.80f);
	colors[ImGuiCol_TabActive] = ImVec4(0.13f, 0.75f, 1.00f, 0.80f);
	colors[ImGuiCol_TabUnfocused] = ImVec4(0.18f, 0.18f, 0.18f, 1.00f);
	colors[ImGuiCol_TabUnfocusedActive] = ImVec4(0.36f, 0.36f, 0.36f, 0.54f);
	colors[ImGuiCol_DockingPreview] = ImVec4(0.13f, 0.75f, 0.55f, 0.80f);
	colors[ImGuiCol_DockingEmptyBg] = ImVec4(0.13f, 0.13f, 0.13f, 0.80f);
	colors[ImGuiCol_PlotLines] = ImVec4(0.61f, 0.61f, 0.61f, 1.00f);
	colors[ImGuiCol_PlotLinesHovered] = ImVec4(1.00f, 0.43f, 0.35f, 1.00f);
	colors[ImGuiCol_PlotHistogram] = ImVec4(0.90f, 0.70f, 0.00f, 1.00f);
	colors[ImGuiCol_PlotHistogramHovered] = ImVec4(1.00f, 0.60f, 0.00f, 1.00f);
	colors[ImGuiCol_TableHeaderBg] = ImVec4(0.19f, 0.19f, 0.20f, 1.00f);
	colors[ImGuiCol_TableBorderStrong] = ImVec4(0.31f, 0.31f, 0.35f, 1.00f);
	colors[ImGuiCol_TableBorderLight] = ImVec4(0.23f, 0.23f, 0.25f, 1.00f);
	colors[ImGuiCol_TableRowBg] = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
	colors[ImGuiCol_TableRowBgAlt] = ImVec4(1.00f, 1.00f, 1.00f, 0.07f);
	colors[ImGuiCol_TextSelectedBg] = ImVec4(0.26f, 0.59f, 0.98f, 0.35f);
	colors[ImGuiCol_DragDropTarget] = ImVec4(1.00f, 1.00f, 0.00f, 0.90f);
	colors[ImGuiCol_NavHighlight] = ImVec4(0.26f, 0.59f, 0.98f, 1.00f);
	colors[ImGuiCol_NavWindowingHighlight] = ImVec4(1.00f, 1.00f, 1.00f, 0.70f);
	colors[ImGuiCol_NavWindowingDimBg] = ImVec4(0.80f, 0.80f, 0.80f, 0.20f);
	colors[ImGuiCol_ModalWindowDimBg] = ImVec4(0.80f, 0.80f, 0.80f, 0.35f);
}
void ImGui::StyleColorsRedDark(ImGuiStyle* dst)
{
	ImGuiStyle* style = dst ? dst : &ImGui::GetStyle();
	ImVec4* colors = style->Colors;

	colors[ImGuiCol_Text] = ImVec4(0.75f, 0.75f, 0.75f, 1.00f);
	colors[ImGuiCol_TextDisabled] = ImVec4(0.35f, 0.35f, 0.35f, 1.00f);
	colors[ImGuiCol_WindowBg] = ImVec4(0.00f, 0.00f, 0.00f, 0.94f);
	colors[ImGuiCol_ChildBg] = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
	colors[ImGuiCol_PopupBg] = ImVec4(0.08f, 0.08f, 0.08f, 0.94f);
	colors[ImGuiCol_Border] = ImVec4(0.00f, 0.00f, 0.00f, 0.50f);
	colors[ImGuiCol_BorderShadow] = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
	colors[ImGuiCol_FrameBg] = ImVec4(0.00f, 0.00f, 0.00f, 0.54f);
	colors[ImGuiCol_FrameBgHovered] = ImVec4(0.37f, 0.14f, 0.14f, 0.67f);
	colors[ImGuiCol_FrameBgActive] = ImVec4(0.39f, 0.20f, 0.20f, 0.67f);
	colors[ImGuiCol_TitleBg] = ImVec4(0.04f, 0.04f, 0.04f, 1.00f);
	colors[ImGuiCol_TitleBgActive] = ImVec4(0.48f, 0.16f, 0.16f, 1.00f);
	colors[ImGuiCol_TitleBgCollapsed] = ImVec4(0.48f, 0.16f, 0.16f, 1.00f);
	colors[ImGuiCol_MenuBarBg] = ImVec4(0.14f, 0.14f, 0.14f, 1.00f);
	colors[ImGuiCol_ScrollbarBg] = ImVec4(0.02f, 0.02f, 0.02f, 0.53f);
	colors[ImGuiCol_ScrollbarGrab] = ImVec4(0.31f, 0.31f, 0.31f, 1.00f);
	colors[ImGuiCol_ScrollbarGrabHovered] = ImVec4(0.41f, 0.41f, 0.41f, 1.00f);
	colors[ImGuiCol_ScrollbarGrabActive] = ImVec4(0.51f, 0.51f, 0.51f, 1.00f);
	colors[ImGuiCol_CheckMark] = ImVec4(0.56f, 0.10f, 0.10f, 1.00f);
	colors[ImGuiCol_SliderGrab] = ImVec4(1.00f, 0.19f, 0.19f, 0.40f);
	colors[ImGuiCol_SliderGrabActive] = ImVec4(0.89f, 0.00f, 0.19f, 1.00f);
	colors[ImGuiCol_Button] = ImVec4(1.00f, 0.19f, 0.19f, 0.40f);
	colors[ImGuiCol_ButtonHovered] = ImVec4(0.80f, 0.17f, 0.00f, 1.00f);
	colors[ImGuiCol_ButtonActive] = ImVec4(0.89f, 0.00f, 0.19f, 1.00f);
	colors[ImGuiCol_Header] = ImVec4(0.33f, 0.35f, 0.36f, 0.53f);
	colors[ImGuiCol_HeaderHovered] = ImVec4(0.76f, 0.28f, 0.44f, 0.67f);
	colors[ImGuiCol_HeaderActive] = ImVec4(0.47f, 0.47f, 0.47f, 0.67f);
	colors[ImGuiCol_Separator] = ImVec4(0.32f, 0.32f, 0.32f, 1.00f);
	colors[ImGuiCol_SeparatorHovered] = ImVec4(0.32f, 0.32f, 0.32f, 1.00f);
	colors[ImGuiCol_SeparatorActive] = ImVec4(0.32f, 0.32f, 0.32f, 1.00f);
	colors[ImGuiCol_ResizeGrip] = ImVec4(1.00f, 1.00f, 1.00f, 0.85f);
	colors[ImGuiCol_ResizeGripHovered] = ImVec4(1.00f, 1.00f, 1.00f, 0.60f);
	colors[ImGuiCol_ResizeGripActive] = ImVec4(1.00f, 1.00f, 1.00f, 0.90f);
	colors[ImGuiCol_Tab] = ImVec4(0.07f, 0.07f, 0.07f, 0.51f);
	colors[ImGuiCol_TabHovered] = ImVec4(0.86f, 0.23f, 0.43f, 0.67f);
	colors[ImGuiCol_TabActive] = ImVec4(0.19f, 0.19f, 0.19f, 0.57f);
	colors[ImGuiCol_TabUnfocused] = ImVec4(0.05f, 0.05f, 0.05f, 0.90f);
	colors[ImGuiCol_TabUnfocusedActive] = ImVec4(0.13f, 0.13f, 0.13f, 0.74f);
	colors[ImGuiCol_DockingPreview] = ImVec4(0.47f, 0.47f, 0.47f, 0.47f);
	colors[ImGuiCol_DockingEmptyBg] = ImVec4(0.20f, 0.20f, 0.20f, 1.00f);
	colors[ImGuiCol_PlotLines] = ImVec4(0.61f, 0.61f, 0.61f, 1.00f);
	colors[ImGuiCol_PlotLinesHovered] = ImVec4(1.00f, 0.43f, 0.35f, 1.00f);
	colors[ImGuiCol_PlotHistogram] = ImVec4(0.90f, 0.70f, 0.00f, 1.00f);
	colors[ImGuiCol_PlotHistogramHovered] = ImVec4(1.00f, 0.60f, 0.00f, 1.00f);
	colors[ImGuiCol_TableHeaderBg] = ImVec4(0.19f, 0.19f, 0.20f, 1.00f);
	colors[ImGuiCol_TableBorderStrong] = ImVec4(0.31f, 0.31f, 0.35f, 1.00f);
	colors[ImGuiCol_TableBorderLight] = ImVec4(0.23f, 0.23f, 0.25f, 1.00f);
	colors[ImGuiCol_TableRowBg] = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
	colors[ImGuiCol_TableRowBgAlt] = ImVec4(1.00f, 1.00f, 1.00f, 0.07f);
	colors[ImGuiCol_TextSelectedBg] = ImVec4(0.26f, 0.59f, 0.98f, 0.35f);
	colors[ImGuiCol_DragDropTarget] = ImVec4(1.00f, 1.00f, 0.00f, 0.90f);
	colors[ImGuiCol_NavHighlight] = ImVec4(0.26f, 0.59f, 0.98f, 1.00f);
	colors[ImGuiCol_NavWindowingHighlight] = ImVec4(1.00f, 1.00f, 1.00f, 0.70f);
	colors[ImGuiCol_NavWindowingDimBg] = ImVec4(0.80f, 0.80f, 0.80f, 0.20f);
	colors[ImGuiCol_ModalWindowDimBg] = ImVec4(0.80f, 0.80f, 0.80f, 0.35f);
}

void ImGui::StyleColorsDeepDark(ImGuiStyle* dst)
{
	ImGuiStyle* style = dst ? dst : &ImGui::GetStyle();
	ImVec4* colors = style->Colors;

	style->WindowPadding = ImVec2(8.00f, 8.00f);
	style->FramePadding = ImVec2(5.00f, 2.00f);
	style->CellPadding = ImVec2(6.00f, 6.00f);
	style->ItemSpacing = ImVec2(6.00f, 6.00f);
	style->ItemInnerSpacing = ImVec2(6.00f, 6.00f);
	style->TouchExtraPadding = ImVec2(0.00f, 0.00f);
	style->IndentSpacing = 25;
	style->ScrollbarSize = 15;
	style->GrabMinSize = 10;
	style->WindowBorderSize = 1;
	style->ChildBorderSize = 1;
	style->PopupBorderSize = 1;
	style->FrameBorderSize = 1;
	style->TabBorderSize = 1;
	style->WindowRounding = 7;
	style->ChildRounding = 4;
	style->FrameRounding = 3;
	style->PopupRounding = 4;
	style->ScrollbarRounding = 9;
	style->GrabRounding = 3;
	style->LogSliderDeadzone = 4;
	style->TabRounding = 4;

	colors[ImGuiCol_Text] = ImVec4(1.00f, 1.00f, 1.00f, 1.00f);
	colors[ImGuiCol_TextDisabled] = ImVec4(0.50f, 0.50f, 0.50f, 1.00f);
	colors[ImGuiCol_WindowBg] = ImVec4(0.10f, 0.10f, 0.10f, 1.00f);
	colors[ImGuiCol_ChildBg] = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
	colors[ImGuiCol_PopupBg] = ImVec4(0.19f, 0.19f, 0.19f, 0.92f);
	colors[ImGuiCol_Border] = ImVec4(0.19f, 0.19f, 0.19f, 0.29f);
	colors[ImGuiCol_BorderShadow] = ImVec4(0.00f, 0.00f, 0.00f, 0.24f);
	colors[ImGuiCol_FrameBg] = ImVec4(0.05f, 0.05f, 0.05f, 0.54f);
	colors[ImGuiCol_FrameBgHovered] = ImVec4(0.19f, 0.19f, 0.19f, 0.54f);
	colors[ImGuiCol_FrameBgActive] = ImVec4(0.20f, 0.22f, 0.23f, 1.00f);
	colors[ImGuiCol_TitleBg] = ImVec4(0.00f, 0.00f, 0.00f, 1.00f);
	colors[ImGuiCol_TitleBgActive] = ImVec4(0.06f, 0.06f, 0.06f, 1.00f);
	colors[ImGuiCol_TitleBgCollapsed] = ImVec4(0.00f, 0.00f, 0.00f, 1.00f);
	colors[ImGuiCol_MenuBarBg] = ImVec4(0.14f, 0.14f, 0.14f, 1.00f);
	colors[ImGuiCol_ScrollbarBg] = ImVec4(0.05f, 0.05f, 0.05f, 0.54f);
	colors[ImGuiCol_ScrollbarGrab] = ImVec4(0.34f, 0.34f, 0.34f, 0.54f);
	colors[ImGuiCol_ScrollbarGrabHovered] = ImVec4(0.40f, 0.40f, 0.40f, 0.54f);
	colors[ImGuiCol_ScrollbarGrabActive] = ImVec4(0.56f, 0.56f, 0.56f, 0.54f);
	colors[ImGuiCol_CheckMark] = ImVec4(0.33f, 0.67f, 0.86f, 1.00f);
	colors[ImGuiCol_SliderGrab] = ImVec4(0.34f, 0.34f, 0.34f, 0.54f);
	colors[ImGuiCol_SliderGrabActive] = ImVec4(0.56f, 0.56f, 0.56f, 0.54f);
	colors[ImGuiCol_Button] = ImVec4(0.05f, 0.05f, 0.05f, 0.54f);
	colors[ImGuiCol_ButtonHovered] = ImVec4(0.19f, 0.19f, 0.19f, 0.54f);
	colors[ImGuiCol_ButtonActive] = ImVec4(0.20f, 0.22f, 0.23f, 1.00f);
	colors[ImGuiCol_Header] = ImVec4(0.00f, 0.00f, 0.00f, 0.52f);
	colors[ImGuiCol_HeaderHovered] = ImVec4(0.00f, 0.00f, 0.00f, 0.36f);
	colors[ImGuiCol_HeaderActive] = ImVec4(0.20f, 0.22f, 0.23f, 0.33f);
	colors[ImGuiCol_Separator] = ImVec4(0.28f, 0.28f, 0.28f, 0.29f);
	colors[ImGuiCol_SeparatorHovered] = ImVec4(0.44f, 0.44f, 0.44f, 0.29f);
	colors[ImGuiCol_SeparatorActive] = ImVec4(0.40f, 0.44f, 0.47f, 1.00f);
	colors[ImGuiCol_ResizeGrip] = ImVec4(0.28f, 0.28f, 0.28f, 0.29f);
	colors[ImGuiCol_ResizeGripHovered] = ImVec4(0.44f, 0.44f, 0.44f, 0.29f);
	colors[ImGuiCol_ResizeGripActive] = ImVec4(0.40f, 0.44f, 0.47f, 1.00f);
	colors[ImGuiCol_Tab] = ImVec4(0.00f, 0.00f, 0.00f, 0.52f);
	colors[ImGuiCol_TabHovered] = ImVec4(0.14f, 0.14f, 0.14f, 1.00f);
	colors[ImGuiCol_TabActive] = ImVec4(0.20f, 0.20f, 0.20f, 0.36f);
	colors[ImGuiCol_TabUnfocused] = ImVec4(0.00f, 0.00f, 0.00f, 0.52f);
	colors[ImGuiCol_TabUnfocusedActive] = ImVec4(0.14f, 0.14f, 0.14f, 1.00f);
	colors[ImGuiCol_DockingPreview] = ImVec4(0.33f, 0.67f, 0.86f, 1.00f);
	colors[ImGuiCol_DockingEmptyBg] = ImVec4(1.00f, 0.00f, 0.00f, 1.00f);
	colors[ImGuiCol_PlotLines] = ImVec4(1.00f, 0.00f, 0.00f, 1.00f);
	colors[ImGuiCol_PlotLinesHovered] = ImVec4(1.00f, 0.00f, 0.00f, 1.00f);
	colors[ImGuiCol_PlotHistogram] = ImVec4(1.00f, 0.00f, 0.00f, 1.00f);
	colors[ImGuiCol_PlotHistogramHovered] = ImVec4(1.00f, 0.00f, 0.00f, 1.00f);
	colors[ImGuiCol_TableHeaderBg] = ImVec4(0.00f, 0.00f, 0.00f, 0.52f);
	colors[ImGuiCol_TableBorderStrong] = ImVec4(0.00f, 0.00f, 0.00f, 0.52f);
	colors[ImGuiCol_TableBorderLight] = ImVec4(0.28f, 0.28f, 0.28f, 0.29f);
	colors[ImGuiCol_TableRowBg] = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
	colors[ImGuiCol_TableRowBgAlt] = ImVec4(1.00f, 1.00f, 1.00f, 0.06f);
	colors[ImGuiCol_TextSelectedBg] = ImVec4(0.20f, 0.22f, 0.23f, 1.00f);
	colors[ImGuiCol_DragDropTarget] = ImVec4(0.33f, 0.67f, 0.86f, 1.00f);
	colors[ImGuiCol_NavHighlight] = ImVec4(1.00f, 0.00f, 0.00f, 1.00f);
	colors[ImGuiCol_NavWindowingHighlight] = ImVec4(1.00f, 0.00f, 0.00f, 0.70f);
	colors[ImGuiCol_NavWindowingDimBg] = ImVec4(1.00f, 0.00f, 0.00f, 0.20f);
	colors[ImGuiCol_ModalWindowDimBg] = ImVec4(1.00f, 0.00f, 0.00f, 0.35f);
}

void ImGui::SetStyle(ImGuiStyle* src)
{
	ImGuiStyle* style = &ImGui::GetStyle();

	style->WindowPadding = src->WindowPadding;
	style->FramePadding = src->FramePadding;
	style->CellPadding = src->CellPadding;
	style->ItemSpacing = src->ItemSpacing;
	style->ItemInnerSpacing = src->ItemInnerSpacing;
	style->TouchExtraPadding = src->TouchExtraPadding;
	style->IndentSpacing = src->IndentSpacing;
	style->ScrollbarSize = src->ScrollbarSize;
	style->GrabMinSize = src->GrabMinSize;
	style->WindowBorderSize = src->WindowBorderSize;
	style->ChildBorderSize = src->ChildBorderSize;
	style->PopupBorderSize = src->PopupBorderSize;
	style->FrameBorderSize = src->FrameBorderSize;
	style->TabBorderSize = src->TabBorderSize;
	style->WindowRounding = src->WindowRounding;
	style->ChildRounding = src->ChildRounding;
	style->FrameRounding = src->FrameRounding;
	style->PopupRounding = src->PopupRounding;
	style->ScrollbarRounding = src->ScrollbarRounding;
	style->GrabRounding = src->GrabRounding;
	style->LogSliderDeadzone = src->LogSliderDeadzone;
	style->TabRounding = src->TabRounding;

	for (int i = 0; i < (int)ImGuiCol_COUNT; i++)
	{
		style->Colors[i] = src->Colors[i];
	}
}

#include <fstream>

const char* ImGuiColToString(ImGuiCol col)
{
	switch (col)
	{
	case ImGuiCol_Text:
		return "ImGuiCol_Text";
	case ImGuiCol_TextDisabled:
		return "ImGuiCol_TextDisabled";
	case ImGuiCol_WindowBg:
		return "ImGuiCol_WindowBg";
	case ImGuiCol_ChildBg:
		return "ImGuiCol_ChildBg";
	case ImGuiCol_PopupBg:
		return "ImGuiCol_PopupBg";
	case ImGuiCol_Border:
		return "ImGuiCol_Border";
	case ImGuiCol_BorderShadow:
		return "ImGuiCol_BorderShadow";
	case ImGuiCol_FrameBg:
		return "ImGuiCol_FrameBg";
	case ImGuiCol_FrameBgHovered:
		return "ImGuiCol_FrameBgHovered";
	case ImGuiCol_FrameBgActive:
		return "ImGuiCol_FrameBgActive";
	case ImGuiCol_TitleBg:
		return "ImGuiCol_TitleBg";
	case ImGuiCol_TitleBgActive:
		return "ImGuiCol_TitleBgActive";
	case ImGuiCol_TitleBgCollapsed:
		return "ImGuiCol_TitleBgCollapsed";
	case ImGuiCol_MenuBarBg:
		return "ImGuiCol_MenuBarBg";
	case ImGuiCol_ScrollbarBg:
		return "ImGuiCol_ScrollbarBg";
	case ImGuiCol_ScrollbarGrab:
		return "ImGuiCol_ScrollbarGrab";
	case ImGuiCol_ScrollbarGrabHovered:
		return "ImGuiCol_ScrollbarGrabHovered";
	case ImGuiCol_ScrollbarGrabActive:
		return "ImGuiCol_ScrollbarGrabActive";
	case ImGuiCol_CheckMark:
		return "ImGuiCol_CheckMark";
	case ImGuiCol_SliderGrab:
		return "ImGuiCol_SliderGrab";
	case ImGuiCol_SliderGrabActive:
		return "ImGuiCol_SliderGrabActive";
	case ImGuiCol_Button:
		return "ImGuiCol_Button";
	case ImGuiCol_ButtonHovered:
		return "ImGuiCol_ButtonHovered";
	case ImGuiCol_ButtonActive:
		return "ImGuiCol_ButtonActive";
	case ImGuiCol_Header:
		return "ImGuiCol_Header";
	case ImGuiCol_HeaderHovered:
		return "ImGuiCol_HeaderHovered";
	case ImGuiCol_HeaderActive:
		return "ImGuiCol_HeaderActive";
	case ImGuiCol_Separator:
		return "ImGuiCol_Separator";
	case ImGuiCol_SeparatorHovered:
		return "ImGuiCol_SeparatorHovered";
	case ImGuiCol_SeparatorActive:
		return "ImGuiCol_SeparatorActive";
	case ImGuiCol_ResizeGrip:
		return "ImGuiCol_ResizeGrip";
	case ImGuiCol_ResizeGripHovered:
		return "ImGuiCol_ResizeGripHovered";
	case ImGuiCol_ResizeGripActive:
		return "ImGuiCol_ResizeGripActive";
	case ImGuiCol_Tab:
		return "ImGuiCol_Tab";
	case ImGuiCol_TabHovered:
		return "ImGuiCol_TabHovered";
	case ImGuiCol_TabActive:
		return "ImGuiCol_TabActive";
	case ImGuiCol_TabUnfocused:
		return "ImGuiCol_TabUnfocused";
	case ImGuiCol_TabUnfocusedActive:
		return "ImGuiCol_TabUnfocusedActive";
	case ImGuiCol_DockingPreview:
		return "ImGuiCol_DockingPreview";
	case ImGuiCol_DockingEmptyBg:
		return "ImGuiCol_DockingEmptyBg";
	case ImGuiCol_PlotLines:
		return "ImGuiCol_PlotLines";
	case ImGuiCol_PlotLinesHovered:
		return "ImGuiCol_PlotLinesHovered";
	case ImGuiCol_PlotHistogram:
		return "ImGuiCol_PlotHistogram";
	case ImGuiCol_PlotHistogramHovered:
		return "ImGuiCol_PlotHistogramHovered";
	case ImGuiCol_TableHeaderBg:
		return "ImGuiCol_TableHeaderBg";
	case ImGuiCol_TableBorderStrong:
		return "ImGuiCol_TableBorderStrong";
	case ImGuiCol_TableBorderLight:
		return "ImGuiCol_TableBorderLight";
	case ImGuiCol_TableRowBg:
		return "ImGuiCol_TableRowBg";
	case ImGuiCol_TableRowBgAlt:
		return "ImGuiCol_TableRowBgAlt";
	case ImGuiCol_TextSelectedBg:
		return "ImGuiCol_TextSelectedBg";
	case ImGuiCol_DragDropTarget:
		return "ImGuiCol_DragDropTarget";
	case ImGuiCol_NavHighlight:
		return "ImGuiCol_NavHighlight";
	case ImGuiCol_NavWindowingHighlight:
		return "ImGuiCol_NavWindowingHighlight";
	case ImGuiCol_NavWindowingDimBg:
		return "ImGuiCol_NavWindowingDimBg";
	case ImGuiCol_ModalWindowDimBg:
		return "ImGuiCol_ModalWindowDimBg";
	}

	return "";
}

static bool saveStyle(const char* filename, EditorStyle currentStyle, const ImGuiStyle* styles, int styleCount)
{
	std::ofstream file(filename, std::ios::trunc);

	if (file.is_open())
	{
		// Write imgui version to file
		file << "IMGUI_VERSION: " << IMGUI_VERSION << "\n";
		file << "Current_Style: " << static_cast<int>(currentStyle) << "\n";

		for (int s = 0; s < styleCount; s++)
		{
			// Write style to file
			file << "Style: " << EditorStyleToString(static_cast<EditorStyle>(s)) << "\n";

			// Write all colors to file
			for (int i = 0; i < (int)ImGuiCol_COUNT; i++)
			{
				file << ImGuiColToString(static_cast<ImGuiCol>(i));
				file << ": ";
				file << styles[s].Colors[i].x << " " << styles[s].Colors[i].y << " " << styles[s].Colors[i].z << " " << styles[s].Colors[i].w;
				file << "\n";
			}

			// Write all floats to file
			file << "Alpha: " << styles[s].Alpha << "\n";
			file << "DisabledAlpha: " << styles[s].DisabledAlpha << "\n";
			file << "WindowRounding: " << styles[s].WindowRounding << "\n";
			file << "WindowBorderSize: " << styles[s].WindowBorderSize << " \n";
			file << "ChildRounding: " << styles[s].ChildRounding << "\n";
			file << "ChildBorderSize: " << styles[s].ChildBorderSize << "\n";
			file << "PopupRounding: " << styles[s].PopupRounding << "\n";
			file << "PopupBorderSize: " << styles[s].PopupBorderSize << "\n";
			file << "FrameRounding: " << styles[s].FrameRounding << "\n";
			file << "FrameBorderSize: " << styles[s].FrameBorderSize << "\n";
			file << "IndentSpacing: " << styles[s].IndentSpacing << "\n";
			file << "ColumnsMinSpacing: " << styles[s].ColumnsMinSpacing << "\n";
			file << "ScrollbarSize: " << styles[s].ScrollbarSize << "\n";
			file << "ScrollbarRounding: " << styles[s].ScrollbarRounding << "\n";
			file << "GrabMinSize: " << styles[s].GrabMinSize << "\n";
			file << "GrabRounding: " << styles[s].GrabRounding << "\n";
			file << "LogSliderDeadzone: " << styles[s].LogSliderDeadzone << "\n";
			file << "TabRounding: " << styles[s].TabRounding << "\n";
			file << "TabBorderSize: " << styles[s].TabBorderSize << "\n";
			file << "TabMinWidthForCloseButton: " << styles[s].TabMinWidthForCloseButton << "\n";
			file << "SeparatorTextBorderSize: " << styles[s].SeparatorTextBorderSize << "\n";
			file << "MouseCursorScale: " << styles[s].MouseCursorScale << "\n";
			file << "CurveTessellationTol: " << styles[s].CurveTessellationTol << "\n";
			file << "CircleTessellationMaxError: " << styles[s].CircleTessellationMaxError << "\n";

			// Write all ImVec2's to file
			file << "WindowPadding: " << styles[s].WindowPadding.x << " " << styles[s].WindowPadding.y << "\n";
			file << "WindowMinSize: " << styles[s].WindowMinSize.x << " " << styles[s].WindowMinSize.y << "\n";
			file << "WindowTitleAlign: " << styles[s].WindowTitleAlign.x << " " << styles[s].WindowTitleAlign.y << "\n";
			file << "FramePadding: " << styles[s].FramePadding.x << " " << styles[s].FramePadding.y << "\n";
			file << "ItemSpacing: " << styles[s].ItemSpacing.x << " " << styles[s].ItemSpacing.y << "\n";
			file << "ItemInnerSpacing: " << styles[s].ItemInnerSpacing.x << " " << styles[s].ItemInnerSpacing.y << "\n";
			file << "CellPadding: " << styles[s].CellPadding.x << " " << styles[s].CellPadding.y << "\n";
			file << "TouchExtraPadding: " << styles[s].TouchExtraPadding.x << " " << styles[s].TouchExtraPadding.y << "\n";
			file << "ButtonTextAlign: " << styles[s].ButtonTextAlign.x << " " << styles[s].ButtonTextAlign.y << "\n";
			file << "SelectableTextAlign: " << styles[s].SelectableTextAlign.x << " " << styles[s].SelectableTextAlign.y << "\n";
			file << "SeparatorTextAlign: " << styles[s].SeparatorTextAlign.x << " " << styles[s].SeparatorTextAlign.y << "\n";
			file << "SeparatorTextPadding: " << styles[s].SeparatorTextPadding.x << " " << styles[s].SeparatorTextPadding.y << "\n";
			file << "DisplayWindowPadding: " << styles[s].DisplayWindowPadding.x << " " << styles[s].DisplayWindowPadding.y << "\n";
			file << "DisplaySafeAreaPadding: " << styles[s].DisplaySafeAreaPadding.x << " " << styles[s].DisplaySafeAreaPadding.y << "\n";

			//ImGuiDir    WindowMenuButtonPosition;
			//ImGuiDir    ColorButtonPosition;
			//bool        AntiAliasedLines;
			//bool        AntiAliasedLinesUseTex;
			//bool        AntiAliasedFill;      
		}

		file.close();
		return true;
	}
	else
	{
		return false;
	}
}

static std::string extractString(const std::string& line, bool& error)
{
	int space = -1; // index of first space
	for (int i = 0; i < line.length(); i++)
	{
		if (line[i] == ' ')
		{
			space = i;
			break;
		}
	}

	if (space == -1)
	{
		// No space found
		error |= true;
		return "";
	}

	// Skip space and go to next character
	space++;

	if (space == line.length())
	{
		// No data after space
		error |= true;
		return "";
	}

	if (line[space] == ' ')
	{
		// Double space detected
		error |= true;
		return "";
	}

	return std::string(&line[space]);
}

static float extractFloat(const std::string& line, bool& error)
{
	int space = -1; // index of first space
	for (int i = 0; i < line.length(); i++)
	{
		if (line[i] == ' ')
		{
			space = i;
			break;
		}
	}

	if (space == -1)
	{
		// No space found
		error |= true;
		return 0.0f;
	}

	// Skip space and go to next character
	space++;

	if (space == line.length())
	{
		// No data after space
		error |= true;
		return 0.0f;
	}

	if (line[space] == ' ')
	{
		// Double space detected
		error |= true;
		return 0.0f;
	}

	return (float)atof(&line[space]);
}

static ImVec2 extractVec2(const std::string& line, bool& error)
{
	int space = -1; // index of first space
	for (int i = 0; i < line.length(); i++)
	{
		if (line[i] == ' ')
		{
			space = i;
			break;
		}
	}

	if (space == -1)
	{
		// No space found
		error |= true;
		return ImVec2(0.0f, 0.0f);
	}

	// Skip space and go to next character
	space++;

	if (space == line.length())
	{
		// No data after space
		error |= true;
		return ImVec2(0.0f, 0.0f);
	}

	if (line[space] == ' ')
	{
		// Double space detected
		error |= true;
		return ImVec2(0.0f, 0.0f);
	}

	if (line.length() - space > 32)
	{
		// Something wrong with the file
		error |= true;
		return ImVec2(0.0f, 0.0f);
	}

	char temp[32];
	memset(temp, '\0', 32);

	int start[2];
	memset(start, 0, 2);

	int j = 0;
	bool flag = true;
	for (int i = space; i < line.length(); i++)
	{
		assert((i - space) < 32);

		if (line[i] != ' ')
		{
			if (flag)
			{
				start[j] = i - space;
				j++;
				flag = false;
			}
			temp[i - space] = line[i];
		}
		else
		{
			temp[i - space] = '\0';
			flag = true;
		}
	}

	if (j != 2)
	{
		// ImVec2 always have exactly 2 components
		error |= true;
		return ImVec2(0.0f, 0.0f);
	}

	ImVec2 vec;
	vec.x = (float)atof(&temp[start[0]]);
	vec.y = (float)atof(&temp[start[1]]);

	return vec;
}

static ImVec4 extractVec4(const std::string& line, bool& error)
{
	int space = -1; // index of first space
	for (int i = 0; i < line.length(); i++)
	{
		if (line[i] == ' ')
		{
			space = i;
			break;
		}
	}

	if (space == -1)
	{
		// No space found
		error |= true;
		return ImVec4(0.0f, 0.0f, 0.0f, 1.0f);
	}

	// Skip space and go to next character
	space++;

	if (space == line.length())
	{
		// No data after space
		error |= true;
		return ImVec4(0.0f, 0.0f, 0.0f, 1.0f);
	}

	if (line[space] == ' ')
	{
		// Double space detected
		error |= true;
		return ImVec4(0.0f, 0.0f, 0.0f, 1.0f);
	}

	if (line.length() - space > 64)
	{
		// Something wrong with the file
		error |= true;
		return ImVec4(0.0f, 0.0f, 0.0f, 1.0f);
	}

	char temp[64];
	memset(temp, '\0', 64);

	int start[4];
	memset(start, 0, 4);

	int j = 0;
	bool flag = true;
	for (int i = space; i < line.length(); i++)
	{
		assert((i - space) < 64);

		if (line[i] != ' ')
		{
			if (flag)
			{
				start[j] = i - space;
				j++;
				flag = false;
			}
			temp[i - space] = line[i];
		}
		else
		{
			temp[i - space] = '\0';
			flag = true;
		}
	}

	if (j != 4)
	{
		// ImVec4 always have exactly 4 components
		error |= true;
		return ImVec4(0.0f, 0.0f, 0.0f, 1.0f);
	}

	ImVec4 vec;
	vec.x = (float)atof(&temp[start[0]]);
	vec.y = (float)atof(&temp[start[1]]);
	vec.z = (float)atof(&temp[start[2]]);
	vec.w = (float)atof(&temp[start[3]]);

	return vec;
}

static bool loadStyle(const char* filename, EditorStyle* currentStyle, ImGuiStyle** styles, int styleCount)
{
	std::ifstream file(filename);

	if (file.is_open())
	{
		int lineNumber = 0;
		std::string line;

		std::string version;
		if (std::getline(file, line))
		{
			bool error = false;
			version = extractString(line, error);
			if (error)
			{
				file.close();
				return false;
			}
			lineNumber++;
		}
		else
		{
			file.close();
			return false;
		}

		if (std::getline(file, line))
		{
			bool error = false;
			*currentStyle = static_cast<EditorStyle>((int)extractFloat(line, error));
			if (error)
			{
				file.close();
				return false;
			}
			lineNumber++;
		}
		else
		{
			file.close();
			return false;
		}

		for (int s = 0; s < styleCount; s++)
		{
			std::string style;
			if (std::getline(file, line))
			{
				bool error = false;
				style = extractString(line, error);
				if (error)
				{
					file.close();
					return false;
				}
				lineNumber++;
			}
			else
			{
				file.close();
				return false;
			}

			int colorCount = 0;
			while (std::getline(file, line))
			{
				if (colorCount < (int)ImGuiCol_COUNT)
				{
					bool error = false;
					(*styles)[s].Colors[colorCount] = extractVec4(line, error);

					if (error)
					{
						return false;
					}

					colorCount++;
					lineNumber++;

					if (colorCount == (int)ImGuiCol_COUNT)
					{
						break;
					}
				}
			}

#if IMGUI_VERSION_NUM == 18971
			std::vector<std::string> nonColorLines;
			nonColorLines.reserve(38);

			int count = 0;
			while (std::getline(file, line))
			{
				nonColorLines.push_back(line);
				count++;
				lineNumber++;

				if (count == 38)
				{
					break;
				}
			}

			if (nonColorLines.size() != 38)
			{
				file.close();
				return false;
			}

			bool error = false;
			(*styles)[s].Alpha = extractFloat(nonColorLines[0], error);
			(*styles)[s].DisabledAlpha = extractFloat(nonColorLines[1], error);
			(*styles)[s].WindowRounding = extractFloat(nonColorLines[2], error);
			(*styles)[s].WindowBorderSize = extractFloat(nonColorLines[3], error);
			(*styles)[s].ChildRounding = extractFloat(nonColorLines[4], error);
			(*styles)[s].ChildBorderSize = extractFloat(nonColorLines[5], error);
			(*styles)[s].PopupRounding = extractFloat(nonColorLines[6], error);
			(*styles)[s].PopupBorderSize = extractFloat(nonColorLines[7], error);
			(*styles)[s].FrameRounding = extractFloat(nonColorLines[8], error);
			(*styles)[s].FrameBorderSize = extractFloat(nonColorLines[9], error);
			(*styles)[s].IndentSpacing = extractFloat(nonColorLines[10], error);
			(*styles)[s].ColumnsMinSpacing = extractFloat(nonColorLines[11], error);
			(*styles)[s].ScrollbarSize = extractFloat(nonColorLines[12], error);
			(*styles)[s].ScrollbarRounding = extractFloat(nonColorLines[13], error);
			(*styles)[s].GrabMinSize = extractFloat(nonColorLines[14], error);
			(*styles)[s].GrabRounding = extractFloat(nonColorLines[15], error);
			(*styles)[s].LogSliderDeadzone = extractFloat(nonColorLines[16], error);
			(*styles)[s].TabRounding = extractFloat(nonColorLines[17], error);
			(*styles)[s].TabBorderSize = extractFloat(nonColorLines[18], error);
			(*styles)[s].TabMinWidthForCloseButton = extractFloat(nonColorLines[19], error);
			(*styles)[s].SeparatorTextBorderSize = extractFloat(nonColorLines[20], error);
			(*styles)[s].MouseCursorScale = extractFloat(nonColorLines[21], error);
			(*styles)[s].CurveTessellationTol = extractFloat(nonColorLines[22], error);
			(*styles)[s].CircleTessellationMaxError = extractFloat(nonColorLines[23], error);

			(*styles)[s].WindowPadding = extractVec2(nonColorLines[24], error);
			(*styles)[s].WindowMinSize = extractVec2(nonColorLines[25], error);
			(*styles)[s].WindowTitleAlign = extractVec2(nonColorLines[26], error);
			(*styles)[s].FramePadding = extractVec2(nonColorLines[27], error);
			(*styles)[s].ItemSpacing = extractVec2(nonColorLines[28], error);
			(*styles)[s].ItemInnerSpacing = extractVec2(nonColorLines[29], error);
			(*styles)[s].CellPadding = extractVec2(nonColorLines[30], error);
			(*styles)[s].TouchExtraPadding = extractVec2(nonColorLines[31], error);
			(*styles)[s].ButtonTextAlign = extractVec2(nonColorLines[32], error);
			(*styles)[s].SelectableTextAlign = extractVec2(nonColorLines[33], error);
			(*styles)[s].SeparatorTextAlign = extractVec2(nonColorLines[34], error);
			(*styles)[s].SeparatorTextPadding = extractVec2(nonColorLines[35], error);
			(*styles)[s].DisplayWindowPadding = extractVec2(nonColorLines[36], error);
			(*styles)[s].DisplaySafeAreaPadding = extractVec2(nonColorLines[37], error);
#endif

			if (error)
			{
				file.close();
				return false;
			}
		}

		file.close();
		return true;
	}
	else
	{
		return false;
	}
}