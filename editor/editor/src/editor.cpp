#include "../include/Editor.h"

#include "../include/imgui/imgui.h"
#include "../include/imgui/imgui_impl_win32.h"
#include "../include/imgui/imgui_impl_opengl3.h"
#include "../include/imgui/imgui_internal.h"

using namespace PhysicsEditor;

Editor::Editor()
{
	isInspectorVisible = true;
	isHierarchyVisible = true;
}

Editor::~Editor()
{

}

void Editor::init(HWND window, int width, int height)
{
	// Setup Dear ImGui binding
	IMGUI_CHECKVERSION();
	ImGui::CreateContext();
	ImGuiIO& io = ImGui::GetIO(); (void)io;

	//Init Win32
	ImGui_ImplWin32_Init(window);

	//Init OpenGL Imgui Implementation
	// GL 3.0 + GLSL 130
	const char* glsl_version = "#version 130";
	ImGui_ImplOpenGL3_Init(glsl_version);

	//Set Window bg color
	ImVec4 clear_color = ImVec4(1.000F, 1.000F, 1.000F, 1.0F);

	// Setup style
	ImGui::StyleColorsClassic();
}

void Editor::cleanUp()
{
	// Cleanup
	ImGui_ImplOpenGL3_Shutdown();
	ImGui::DestroyContext();
	ImGui_ImplWin32_Shutdown();
}

static bool open = true;

void Editor::render()
{
	// Start the Dear ImGui frame
	ImGui_ImplOpenGL3_NewFrame();
	ImGui_ImplWin32_NewFrame();
	ImGui::NewFrame();

	ImGui::ShowDemoWindow();
	//ImGui::ShowDemoWindowWidgets();

	mainMenu.render();
	//inspector.render();


	//HelpMarker("BeginGroup() basically locks the horizontal position for new line. EndGroup() bundles the whole group so that you can use \"item\" functions such as IsItemHovered()/IsItemActive() or SameLine() etc. on the whole group.");
	//ImGui::BeginGroup();
	//{
	//	ImGui::BeginGroup();
	//	ImGui::Button("AAA");
	//	ImGui::SameLine();
	//	ImGui::Button("BBB");
	//	ImGui::SameLine();
	//	ImGui::BeginGroup();
	//	ImGui::Button("CCC");
	//	ImGui::Button("DDD");
	//	ImGui::EndGroup();
	//	ImGui::SameLine();
	//	ImGui::Button("EEE");
	//	ImGui::EndGroup();
	//	if (ImGui::IsItemHovered())
	//		ImGui::SetTooltip("First group hovered");
	//}
	//ImGui::SameLine();
	//// Capture the group size and create widgets using the same size
	//ImVec2 size = ImGui::GetItemRectSize();
	//const float values[5] = { 0.5f, 0.20f, 0.80f, 0.60f, 0.25f };
	//ImGui::PlotHistogram("##values", values, IM_ARRAYSIZE(values), 0, NULL, 0.0f, 1.0f, size);

	//ImGui::Button("ACTION", ImVec2((size.x - ImGui::GetStyle().ItemSpacing.x)*0.5f, size.y));
	//ImGui::SameLine();
	//ImGui::Button("REACTION", ImVec2((size.x - ImGui::GetStyle().ItemSpacing.x)*0.5f, size.y));
	//ImGui::EndGroup();


	// Rendering
	ImGui::Render();
	//wglMakeCurrent(deviceContext, renderContext);
	//glViewport(0, 0, g_display_w, g_display_h);                 //Display Size got from Resize Command
	glViewport(0, 0, 200, 200);
	/*glClearColor(clear_color.x, clear_color.y, clear_color.z, clear_color.w);*/
	glClearColor(0.15f, 0.15f, 0.15f, 0.0f);
	glClear(GL_COLOR_BUFFER_BIT);
	ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
	//wglMakeCurrent(deviceContext, renderContext);
	//SwapBuffers(deviceContext);
}

















//void ImGui::ShowDemoWindow(bool* p_open)
//{
//	IM_ASSERT(ImGui::GetCurrentContext() != NULL && "Missing dear imgui context. Refer to examples app!"); // Exceptionally add an extra assert here for people confused with initial dear imgui setup
//
//	// Examples Apps (accessible from the "Examples" menu)
//	static bool show_app_documents = false;
//	static bool show_app_main_menu_bar = false;
//	static bool show_app_console = false;
//	static bool show_app_log = false;
//	static bool show_app_layout = false;
//	static bool show_app_property_editor = false;
//	static bool show_app_long_text = false;
//	static bool show_app_auto_resize = false;
//	static bool show_app_constrained_resize = false;
//	static bool show_app_simple_overlay = false;
//	static bool show_app_window_titles = false;
//	static bool show_app_custom_rendering = false;
//
//	if (show_app_documents)           ShowExampleAppDocuments(&show_app_documents);
//	if (show_app_main_menu_bar)       ShowExampleAppMainMenuBar();
//	if (show_app_console)             ShowExampleAppConsole(&show_app_console);
//	if (show_app_log)                 ShowExampleAppLog(&show_app_log);
//	if (show_app_layout)              ShowExampleAppLayout(&show_app_layout);
//	if (show_app_property_editor)     ShowExampleAppPropertyEditor(&show_app_property_editor);
//	if (show_app_long_text)           ShowExampleAppLongText(&show_app_long_text);
//	if (show_app_auto_resize)         ShowExampleAppAutoResize(&show_app_auto_resize);
//	if (show_app_constrained_resize)  ShowExampleAppConstrainedResize(&show_app_constrained_resize);
//	if (show_app_simple_overlay)      ShowExampleAppSimpleOverlay(&show_app_simple_overlay);
//	if (show_app_window_titles)       ShowExampleAppWindowTitles(&show_app_window_titles);
//	if (show_app_custom_rendering)    ShowExampleAppCustomRendering(&show_app_custom_rendering);
//
//	// Dear ImGui Apps (accessible from the "Help" menu)
//	static bool show_app_metrics = false;
//	static bool show_app_style_editor = false;
//	static bool show_app_about = false;
//
//	if (show_app_metrics)             { ImGui::ShowMetricsWindow(&show_app_metrics); }
//	if (show_app_style_editor)        { ImGui::Begin("Style Editor", &show_app_style_editor); ImGui::ShowStyleEditor(); ImGui::End(); }
//	if (show_app_about)               { ImGui::ShowAboutWindow(&show_app_about); }
//
//	// Demonstrate the various window flags. Typically you would just use the default!
//	static bool no_titlebar = false;
//	static bool no_scrollbar = false;
//	static bool no_menu = false;
//	static bool no_move = false;
//	static bool no_resize = false;
//	static bool no_collapse = false;
//	static bool no_close = false;
//	static bool no_nav = false;
//	static bool no_background = false;
//	static bool no_bring_to_front = false;
//
//	ImGuiWindowFlags window_flags = 0;
//	if (no_titlebar)        window_flags |= ImGuiWindowFlags_NoTitleBar;
//	if (no_scrollbar)       window_flags |= ImGuiWindowFlags_NoScrollbar;
//	if (!no_menu)           window_flags |= ImGuiWindowFlags_MenuBar;
//	if (no_move)            window_flags |= ImGuiWindowFlags_NoMove;
//	if (no_resize)          window_flags |= ImGuiWindowFlags_NoResize;
//	if (no_collapse)        window_flags |= ImGuiWindowFlags_NoCollapse;
//	if (no_nav)             window_flags |= ImGuiWindowFlags_NoNav;
//	if (no_background)      window_flags |= ImGuiWindowFlags_NoBackground;
//	if (no_bring_to_front)  window_flags |= ImGuiWindowFlags_NoBringToFrontOnFocus;
//	if (no_close)           p_open = NULL; // Don't pass our bool* to Begin
//
//	// We specify a default position/size in case there's no data in the .ini file. Typically this isn't required! We only do it to make the Demo applications a little more welcoming.
//	ImGui::SetNextWindowPos(ImVec2(650, 20), ImGuiCond_FirstUseEver);
//	ImGui::SetNextWindowSize(ImVec2(550, 680), ImGuiCond_FirstUseEver);
//
//	// Main body of the Demo window starts here.
//	if (!ImGui::Begin("Dear ImGui Demo", p_open, window_flags))
//	{
//		// Early out if the window is collapsed, as an optimization.
//		ImGui::End();
//		return;
//	}
//
//	// Most "big" widgets share a common width settings by default.
//	//ImGui::PushItemWidth(ImGui::GetWindowWidth() * 0.65f);    // Use 2/3 of the space for widgets and 1/3 for labels (default)
//	ImGui::PushItemWidth(ImGui::GetFontSize() * -12);           // Use fixed width for labels (by passing a negative value), the rest goes to widgets. We choose a width proportional to our font size.
//
//	// Menu Bar
//	if (ImGui::BeginMenuBar())
//	{
//		if (ImGui::BeginMenu("Menu"))
//		{
//			ShowExampleMenuFile();
//			ImGui::EndMenu();
//		}
//		if (ImGui::BeginMenu("Examples"))
//		{
//			ImGui::MenuItem("Main menu bar", NULL, &show_app_main_menu_bar);
//			ImGui::MenuItem("Console", NULL, &show_app_console);
//			ImGui::MenuItem("Log", NULL, &show_app_log);
//			ImGui::MenuItem("Simple layout", NULL, &show_app_layout);
//			ImGui::MenuItem("Property editor", NULL, &show_app_property_editor);
//			ImGui::MenuItem("Long text display", NULL, &show_app_long_text);
//			ImGui::MenuItem("Auto-resizing window", NULL, &show_app_auto_resize);
//			ImGui::MenuItem("Constrained-resizing window", NULL, &show_app_constrained_resize);
//			ImGui::MenuItem("Simple overlay", NULL, &show_app_simple_overlay);
//			ImGui::MenuItem("Manipulating window titles", NULL, &show_app_window_titles);
//			ImGui::MenuItem("Custom rendering", NULL, &show_app_custom_rendering);
//			ImGui::MenuItem("Documents", NULL, &show_app_documents);
//			ImGui::EndMenu();
//		}
//		if (ImGui::BeginMenu("Help"))
//		{
//			ImGui::MenuItem("Metrics", NULL, &show_app_metrics);
//			ImGui::MenuItem("Style Editor", NULL, &show_app_style_editor);
//			ImGui::MenuItem("About Dear ImGui", NULL, &show_app_about);
//			ImGui::EndMenu();
//		}
//		ImGui::EndMenuBar();
//	}
//
//	ImGui::Text("dear imgui says hello. (%s)", IMGUI_VERSION);
//	ImGui::Spacing();
//
//	if (ImGui::CollapsingHeader("Help"))
//	{
//		ImGui::Text("PROGRAMMER GUIDE:");
//		ImGui::BulletText("Please see the ShowDemoWindow() code in imgui_demo.cpp. <- you are here!");
//		ImGui::BulletText("Please see the comments in imgui.cpp.");
//		ImGui::BulletText("Please see the examples/ in application.");
//		ImGui::BulletText("Enable 'io.ConfigFlags |= NavEnableKeyboard' for keyboard controls.");
//		ImGui::BulletText("Enable 'io.ConfigFlags |= NavEnableGamepad' for gamepad controls.");
//		ImGui::Separator();
//
//		ImGui::Text("USER GUIDE:");
//		ImGui::ShowUserGuide();
//	}
//
//	if (ImGui::CollapsingHeader("Configuration"))
//	{
//		ImGuiIO& io = ImGui::GetIO();
//
//		if (ImGui::TreeNode("Configuration##2"))
//		{
//			ImGui::CheckboxFlags("io.ConfigFlags: NavEnableKeyboard", (unsigned int *)&io.ConfigFlags, ImGuiConfigFlags_NavEnableKeyboard);
//			ImGui::CheckboxFlags("io.ConfigFlags: NavEnableGamepad", (unsigned int *)&io.ConfigFlags, ImGuiConfigFlags_NavEnableGamepad);
//			ImGui::SameLine(); HelpMarker("Required back-end to feed in gamepad inputs in io.NavInputs[] and set io.BackendFlags |= ImGuiBackendFlags_HasGamepad.\n\nRead instructions in imgui.cpp for details.");
//			ImGui::CheckboxFlags("io.ConfigFlags: NavEnableSetMousePos", (unsigned int *)&io.ConfigFlags, ImGuiConfigFlags_NavEnableSetMousePos);
//			ImGui::SameLine(); HelpMarker("Instruct navigation to move the mouse cursor. See comment for ImGuiConfigFlags_NavEnableSetMousePos.");
//			ImGui::CheckboxFlags("io.ConfigFlags: NoMouse", (unsigned int *)&io.ConfigFlags, ImGuiConfigFlags_NoMouse);
//			if (io.ConfigFlags & ImGuiConfigFlags_NoMouse) // Create a way to restore this flag otherwise we could be stuck completely!
//			{
//				if (fmodf((float)ImGui::GetTime(), 0.40f) < 0.20f)
//				{
//					ImGui::SameLine();
//					ImGui::Text("<<PRESS SPACE TO DISABLE>>");
//				}
//				if (ImGui::IsKeyPressed(ImGui::GetKeyIndex(ImGuiKey_Space)))
//					io.ConfigFlags &= ~ImGuiConfigFlags_NoMouse;
//			}
//			ImGui::CheckboxFlags("io.ConfigFlags: NoMouseCursorChange", (unsigned int *)&io.ConfigFlags, ImGuiConfigFlags_NoMouseCursorChange);
//			ImGui::SameLine(); HelpMarker("Instruct back-end to not alter mouse cursor shape and visibility.");
//			ImGui::Checkbox("io.ConfigInputTextCursorBlink", &io.ConfigInputTextCursorBlink);
//			ImGui::SameLine(); HelpMarker("Set to false to disable blinking cursor, for users who consider it distracting");
//			ImGui::Checkbox("io.ConfigWindowsResizeFromEdges", &io.ConfigWindowsResizeFromEdges);
//			ImGui::SameLine(); HelpMarker("Enable resizing of windows from their edges and from the lower-left corner.\nThis requires (io.BackendFlags & ImGuiBackendFlags_HasMouseCursors) because it needs mouse cursor feedback.");
//			ImGui::Checkbox("io.ConfigWindowsMoveFromTitleBarOnly", &io.ConfigWindowsMoveFromTitleBarOnly);
//			ImGui::Checkbox("io.MouseDrawCursor", &io.MouseDrawCursor);
//			ImGui::SameLine(); HelpMarker("Instruct Dear ImGui to render a mouse cursor for you. Note that a mouse cursor rendered via your application GPU rendering path will feel more laggy than hardware cursor, but will be more in sync with your other visuals.\n\nSome desktop applications may use both kinds of cursors (e.g. enable software cursor only when resizing/dragging something).");
//			ImGui::TreePop();
//			ImGui::Separator();
//		}
//
//		if (ImGui::TreeNode("Backend Flags"))
//		{
//			HelpMarker("Those flags are set by the back-ends (imgui_impl_xxx files) to specify their capabilities.");
//			ImGuiBackendFlags backend_flags = io.BackendFlags; // Make a local copy to avoid modifying actual back-end flags.
//			ImGui::CheckboxFlags("io.BackendFlags: HasGamepad", (unsigned int *)&backend_flags, ImGuiBackendFlags_HasGamepad);
//			ImGui::CheckboxFlags("io.BackendFlags: HasMouseCursors", (unsigned int *)&backend_flags, ImGuiBackendFlags_HasMouseCursors);
//			ImGui::CheckboxFlags("io.BackendFlags: HasSetMousePos", (unsigned int *)&backend_flags, ImGuiBackendFlags_HasSetMousePos);
//			ImGui::CheckboxFlags("io.BackendFlags: RendererHasVtxOffset", (unsigned int *)&backend_flags, ImGuiBackendFlags_RendererHasVtxOffset);
//			ImGui::TreePop();
//			ImGui::Separator();
//		}
//
//		if (ImGui::TreeNode("Style"))
//		{
//			ImGui::ShowStyleEditor();
//			ImGui::TreePop();
//			ImGui::Separator();
//		}
//
//		if (ImGui::TreeNode("Capture/Logging"))
//		{
//			ImGui::TextWrapped("The logging API redirects all text output so you can easily capture the content of a window or a block. Tree nodes can be automatically expanded.");
//			HelpMarker("Try opening any of the contents below in this window and then click one of the \"Log To\" button.");
//			ImGui::LogButtons();
//			ImGui::TextWrapped("You can also call ImGui::LogText() to output directly to the log without a visual output.");
//			if (ImGui::Button("Copy \"Hello, world!\" to clipboard"))
//			{
//				ImGui::LogToClipboard();
//				ImGui::LogText("Hello, world!");
//				ImGui::LogFinish();
//			}
//			ImGui::TreePop();
//		}
//	}
//
//	if (ImGui::CollapsingHeader("Window options"))
//	{
//		ImGui::Checkbox("No titlebar", &no_titlebar); ImGui::SameLine(150);
//		ImGui::Checkbox("No scrollbar", &no_scrollbar); ImGui::SameLine(300);
//		ImGui::Checkbox("No menu", &no_menu);
//		ImGui::Checkbox("No move", &no_move); ImGui::SameLine(150);
//		ImGui::Checkbox("No resize", &no_resize); ImGui::SameLine(300);
//		ImGui::Checkbox("No collapse", &no_collapse);
//		ImGui::Checkbox("No close", &no_close); ImGui::SameLine(150);
//		ImGui::Checkbox("No nav", &no_nav); ImGui::SameLine(300);
//		ImGui::Checkbox("No background", &no_background);
//		ImGui::Checkbox("No bring to front", &no_bring_to_front);
//	}
//
//	// All demo contents
//	ShowDemoWindowWidgets();
//	ShowDemoWindowLayout();
//	ShowDemoWindowPopups();
//	ShowDemoWindowColumns();
//	ShowDemoWindowMisc();
//
//	// End of ShowDemoWindow()
//	ImGui::End();
//}
//
//static void ShowDemoWindowWidgets()
//{
//	if (!ImGui::CollapsingHeader("Widgets"))
//		return;
//
//	if (ImGui::TreeNode("Basic"))
//	{
//		static int clicked = 0;
//		if (ImGui::Button("Button"))
//			clicked++;
//		if (clicked & 1)
//		{
//			ImGui::SameLine();
//			ImGui::Text("Thanks for clicking me!");
//		}
//
//		static bool check = true;
//		ImGui::Checkbox("checkbox", &check);
//
//		static int e = 0;
//		ImGui::RadioButton("radio a", &e, 0); ImGui::SameLine();
//		ImGui::RadioButton("radio b", &e, 1); ImGui::SameLine();
//		ImGui::RadioButton("radio c", &e, 2);
//
//		// Color buttons, demonstrate using PushID() to add unique identifier in the ID stack, and changing style.
//		for (int i = 0; i < 7; i++)
//		{
//			if (i > 0)
//				ImGui::SameLine();
//			ImGui::PushID(i);
//			ImGui::PushStyleColor(ImGuiCol_Button, (ImVec4)ImColor::HSV(i / 7.0f, 0.6f, 0.6f));
//			ImGui::PushStyleColor(ImGuiCol_ButtonHovered, (ImVec4)ImColor::HSV(i / 7.0f, 0.7f, 0.7f));
//			ImGui::PushStyleColor(ImGuiCol_ButtonActive, (ImVec4)ImColor::HSV(i / 7.0f, 0.8f, 0.8f));
//			ImGui::Button("Click");
//			ImGui::PopStyleColor(3);
//			ImGui::PopID();
//		}
//
//		// Use AlignTextToFramePadding() to align text baseline to the baseline of framed elements (otherwise a Text+SameLine+Button sequence will have the text a little too high by default)
//		ImGui::AlignTextToFramePadding();
//		ImGui::Text("Hold to repeat:");
//		ImGui::SameLine();
//
//		// Arrow buttons with Repeater
//		static int counter = 0;
//		float spacing = ImGui::GetStyle().ItemInnerSpacing.x;
//		ImGui::PushButtonRepeat(true);
//		if (ImGui::ArrowButton("##left", ImGuiDir_Left)) { counter--; }
//		ImGui::SameLine(0.0f, spacing);
//		if (ImGui::ArrowButton("##right", ImGuiDir_Right)) { counter++; }
//		ImGui::PopButtonRepeat();
//		ImGui::SameLine();
//		ImGui::Text("%d", counter);
//
//		ImGui::Text("Hover over me");
//		if (ImGui::IsItemHovered())
//			ImGui::SetTooltip("I am a tooltip");
//
//		ImGui::SameLine();
//		ImGui::Text("- or me");
//		if (ImGui::IsItemHovered())
//		{
//			ImGui::BeginTooltip();
//			ImGui::Text("I am a fancy tooltip");
//			static float arr[] = { 0.6f, 0.1f, 1.0f, 0.5f, 0.92f, 0.1f, 0.2f };
//			ImGui::PlotLines("Curve", arr, IM_ARRAYSIZE(arr));
//			ImGui::EndTooltip();
//		}
//
//		ImGui::Separator();
//
//		ImGui::LabelText("label", "Value");
//
//		{
//			// Using the _simplified_ one-liner Combo() api here
//			// See "Combo" section for examples of how to use the more complete BeginCombo()/EndCombo() api.
//			const char* items[] = { "AAAA", "BBBB", "CCCC", "DDDD", "EEEE", "FFFF", "GGGG", "HHHH", "IIII", "JJJJ", "KKKK", "LLLLLLL", "MMMM", "OOOOOOO" };
//			static int item_current = 0;
//			ImGui::Combo("combo", &item_current, items, IM_ARRAYSIZE(items));
//			ImGui::SameLine(); HelpMarker("Refer to the \"Combo\" section below for an explanation of the full BeginCombo/EndCombo API, and demonstration of various flags.\n");
//		}
//
//		{
//			static char str0[128] = "Hello, world!";
//			ImGui::InputText("input text", str0, IM_ARRAYSIZE(str0));
//			ImGui::SameLine(); HelpMarker("USER:\nHold SHIFT or use mouse to select text.\n" "CTRL+Left/Right to word jump.\n" "CTRL+A or double-click to select all.\n" "CTRL+X,CTRL+C,CTRL+V clipboard.\n" "CTRL+Z,CTRL+Y undo/redo.\n" "ESCAPE to revert.\n\nPROGRAMMER:\nYou can use the ImGuiInputTextFlags_CallbackResize facility if you need to wire InputText() to a dynamic string type. See misc/cpp/imgui_stdlib.h for an example (this is not demonstrated in imgui_demo.cpp).");
//
//			static char str1[128] = "";
//			ImGui::InputTextWithHint("input text (w/ hint)", "enter text here", str1, IM_ARRAYSIZE(str1));
//
//			static int i0 = 123;
//			ImGui::InputInt("input int", &i0);
//			ImGui::SameLine(); HelpMarker("You can apply arithmetic operators +,*,/ on numerical values.\n  e.g. [ 100 ], input \'*2\', result becomes [ 200 ]\nUse +- to subtract.\n");
//
//			static float f0 = 0.001f;
//			ImGui::InputFloat("input float", &f0, 0.01f, 1.0f, "%.3f");
//
//			static double d0 = 999999.00000001;
//			ImGui::InputDouble("input double", &d0, 0.01f, 1.0f, "%.8f");
//
//			static float f1 = 1.e10f;
//			ImGui::InputFloat("input scientific", &f1, 0.0f, 0.0f, "%e");
//			ImGui::SameLine(); HelpMarker("You can input value using the scientific notation,\n  e.g. \"1e+8\" becomes \"100000000\".\n");
//
//			static float vec4a[4] = { 0.10f, 0.20f, 0.30f, 0.44f };
//			ImGui::InputFloat3("input float3", vec4a);
//		}
//
//		{
//			static int i1 = 50, i2 = 42;
//			ImGui::DragInt("drag int", &i1, 1);
//			ImGui::SameLine(); HelpMarker("Click and drag to edit value.\nHold SHIFT/ALT for faster/slower edit.\nDouble-click or CTRL+click to input value.");
//
//			ImGui::DragInt("drag int 0..100", &i2, 1, 0, 100, "%d%%");
//
//			static float f1 = 1.00f, f2 = 0.0067f;
//			ImGui::DragFloat("drag float", &f1, 0.005f);
//			ImGui::DragFloat("drag small float", &f2, 0.0001f, 0.0f, 0.0f, "%.06f ns");
//		}
//
//		{
//			static int i1 = 0;
//			ImGui::SliderInt("slider int", &i1, -1, 3);
//			ImGui::SameLine(); HelpMarker("CTRL+click to input value.");
//
//			static float f1 = 0.123f, f2 = 0.0f;
//			ImGui::SliderFloat("slider float", &f1, 0.0f, 1.0f, "ratio = %.3f");
//			ImGui::SliderFloat("slider float (curve)", &f2, -10.0f, 10.0f, "%.4f", 2.0f);
//			static float angle = 0.0f;
//			ImGui::SliderAngle("slider angle", &angle);
//		}
//
//		{
//			static float col1[3] = { 1.0f, 0.0f, 0.2f };
//			static float col2[4] = { 0.4f, 0.7f, 0.0f, 0.5f };
//			ImGui::ColorEdit3("color 1", col1);
//			ImGui::SameLine(); HelpMarker("Click on the colored square to open a color picker.\nClick and hold to use drag and drop.\nRight-click on the colored square to show options.\nCTRL+click on individual component to input value.\n");
//
//			ImGui::ColorEdit4("color 2", col2);
//		}
//
//		{
//			// List box
//			const char* listbox_items[] = { "Apple", "Banana", "Cherry", "Kiwi", "Mango", "Orange", "Pineapple", "Strawberry", "Watermelon" };
//			static int listbox_item_current = 1;
//			ImGui::ListBox("listbox\n(single select)", &listbox_item_current, listbox_items, IM_ARRAYSIZE(listbox_items), 4);
//
//			//static int listbox_item_current2 = 2;
//			//ImGui::SetNextItemWidth(-1);
//			//ImGui::ListBox("##listbox2", &listbox_item_current2, listbox_items, IM_ARRAYSIZE(listbox_items), 4);
//		}
//
//		ImGui::TreePop();
//	}
//
//	// Testing ImGuiOnceUponAFrame helper.
//	//static ImGuiOnceUponAFrame once;
//	//for (int i = 0; i < 5; i++)
//	//    if (once)
//	//        ImGui::Text("This will be displayed only once.");
//
//	if (ImGui::TreeNode("Trees"))
//	{
//		if (ImGui::TreeNode("Basic trees"))
//		{
//			for (int i = 0; i < 5; i++)
//			{
//				// Use SetNextItemOpen() so set the default state of a node to be open. 
//				// We could also use TreeNodeEx() with the ImGuiTreeNodeFlags_DefaultOpen flag to achieve the same thing!
//				if (i == 0)
//					ImGui::SetNextItemOpen(true, ImGuiCond_Once);
//
//				if (ImGui::TreeNode((void*)(intptr_t)i, "Child %d", i))
//				{
//					ImGui::Text("blah blah");
//					ImGui::SameLine();
//					if (ImGui::SmallButton("button")) {};
//					ImGui::TreePop();
//				}
//			}
//			ImGui::TreePop();
//		}
//
//		if (ImGui::TreeNode("Advanced, with Selectable nodes"))
//		{
//			HelpMarker("This is a more typical looking tree with selectable nodes.\nClick to select, CTRL+Click to toggle, click on arrows or double-click to open.");
//			static bool align_label_with_current_x_position = false;
//			ImGui::Checkbox("Align label with current X position)", &align_label_with_current_x_position);
//			ImGui::Text("Hello!");
//			if (align_label_with_current_x_position)
//				ImGui::Unindent(ImGui::GetTreeNodeToLabelSpacing());
//
//			static int selection_mask = (1 << 2); // Dumb representation of what may be user-side selection state. You may carry selection state inside or outside your objects in whatever format you see fit.
//			int node_clicked = -1;                // Temporary storage of what node we have clicked to process selection at the end of the loop. May be a pointer to your own node type, etc.
//			ImGui::PushStyleVar(ImGuiStyleVar_IndentSpacing, ImGui::GetFontSize() * 3); // Increase spacing to differentiate leaves from expanded contents.
//			for (int i = 0; i < 6; i++)
//			{
//				// Disable the default open on single-click behavior and pass in Selected flag according to our selection state.
//				ImGuiTreeNodeFlags node_flags = ImGuiTreeNodeFlags_OpenOnArrow | ImGuiTreeNodeFlags_OpenOnDoubleClick;
//				if (selection_mask & (1 << i))
//					node_flags |= ImGuiTreeNodeFlags_Selected;
//				if (i < 3)
//				{
//					// Items 0..2 are Tree Node
//					bool node_open = ImGui::TreeNodeEx((void*)(intptr_t)i, node_flags, "Selectable Node %d", i);
//					if (ImGui::IsItemClicked())
//						node_clicked = i;
//					if (node_open)
//					{
//						ImGui::Text("Blah blah\nBlah Blah");
//						ImGui::TreePop();
//					}
//				}
//				else
//				{
//					// Items 3..5 are Tree Leaves
//					// The only reason we use TreeNode at all is to allow selection of the leaf.
//					// Otherwise we can use BulletText() or TreeAdvanceToLabelPos()+Text().
//					node_flags |= ImGuiTreeNodeFlags_Leaf | ImGuiTreeNodeFlags_NoTreePushOnOpen; // ImGuiTreeNodeFlags_Bullet
//					ImGui::TreeNodeEx((void*)(intptr_t)i, node_flags, "Selectable Leaf %d", i);
//					if (ImGui::IsItemClicked())
//						node_clicked = i;
//				}
//			}
//			if (node_clicked != -1)
//			{
//				// Update selection state. Process outside of tree loop to avoid visual inconsistencies during the clicking-frame.
//				if (ImGui::GetIO().KeyCtrl)
//					selection_mask ^= (1 << node_clicked);          // CTRL+click to toggle
//				else //if (!(selection_mask & (1 << node_clicked))) // Depending on selection behavior you want, this commented bit preserve selection when clicking on item that is part of the selection
//					selection_mask = (1 << node_clicked);           // Click to single-select
//			}
//			ImGui::PopStyleVar();
//			if (align_label_with_current_x_position)
//				ImGui::Indent(ImGui::GetTreeNodeToLabelSpacing());
//			ImGui::TreePop();
//		}
//		ImGui::TreePop();
//	}
//
//	if (ImGui::TreeNode("Collapsing Headers"))
//	{
//		static bool closable_group = true;
//		ImGui::Checkbox("Show 2nd header", &closable_group);
//		if (ImGui::CollapsingHeader("Header"))
//		{
//			ImGui::Text("IsItemHovered: %d", ImGui::IsItemHovered());
//			for (int i = 0; i < 5; i++)
//				ImGui::Text("Some content %d", i);
//		}
//		if (ImGui::CollapsingHeader("Header with a close button", &closable_group))
//		{
//			ImGui::Text("IsItemHovered: %d", ImGui::IsItemHovered());
//			for (int i = 0; i < 5; i++)
//				ImGui::Text("More content %d", i);
//		}
//		ImGui::TreePop();
//	}
//
//	if (ImGui::TreeNode("Bullets"))
//	{
//		ImGui::BulletText("Bullet point 1");
//		ImGui::BulletText("Bullet point 2\nOn multiple lines");
//		ImGui::Bullet(); ImGui::Text("Bullet point 3 (two calls)");
//		ImGui::Bullet(); ImGui::SmallButton("Button");
//		ImGui::TreePop();
//	}
//
//	if (ImGui::TreeNode("Text"))
//	{
//		if (ImGui::TreeNode("Colored Text"))
//		{
//			// Using shortcut. You can use PushStyleColor()/PopStyleColor() for more flexibility.
//			ImGui::TextColored(ImVec4(1.0f, 0.0f, 1.0f, 1.0f), "Pink");
//			ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.0f, 1.0f), "Yellow");
//			ImGui::TextDisabled("Disabled");
//			ImGui::SameLine(); HelpMarker("The TextDisabled color is stored in ImGuiStyle.");
//			ImGui::TreePop();
//		}
//
//		if (ImGui::TreeNode("Word Wrapping"))
//		{
//			// Using shortcut. You can use PushTextWrapPos()/PopTextWrapPos() for more flexibility.
//			ImGui::TextWrapped("This text should automatically wrap on the edge of the window. The current implementation for text wrapping follows simple rules suitable for English and possibly other languages.");
//			ImGui::Spacing();
//
//			static float wrap_width = 200.0f;
//			ImGui::SliderFloat("Wrap width", &wrap_width, -20, 600, "%.0f");
//
//			ImGui::Text("Test paragraph 1:");
//			ImVec2 pos = ImGui::GetCursorScreenPos();
//			ImGui::GetWindowDrawList()->AddRectFilled(ImVec2(pos.x + wrap_width, pos.y), ImVec2(pos.x + wrap_width + 10, pos.y + ImGui::GetTextLineHeight()), IM_COL32(255, 0, 255, 255));
//			ImGui::PushTextWrapPos(ImGui::GetCursorPos().x + wrap_width);
//			ImGui::Text("The lazy dog is a good dog. This paragraph is made to fit within %.0f pixels. Testing a 1 character word. The quick brown fox jumps over the lazy dog.", wrap_width);
//			ImGui::GetWindowDrawList()->AddRect(ImGui::GetItemRectMin(), ImGui::GetItemRectMax(), IM_COL32(255, 255, 0, 255));
//			ImGui::PopTextWrapPos();
//
//			ImGui::Text("Test paragraph 2:");
//			pos = ImGui::GetCursorScreenPos();
//			ImGui::GetWindowDrawList()->AddRectFilled(ImVec2(pos.x + wrap_width, pos.y), ImVec2(pos.x + wrap_width + 10, pos.y + ImGui::GetTextLineHeight()), IM_COL32(255, 0, 255, 255));
//			ImGui::PushTextWrapPos(ImGui::GetCursorPos().x + wrap_width);
//			ImGui::Text("aaaaaaaa bbbbbbbb, c cccccccc,dddddddd. d eeeeeeee   ffffffff. gggggggg!hhhhhhhh");
//			ImGui::GetWindowDrawList()->AddRect(ImGui::GetItemRectMin(), ImGui::GetItemRectMax(), IM_COL32(255, 255, 0, 255));
//			ImGui::PopTextWrapPos();
//
//			ImGui::TreePop();
//		}
//
//		if (ImGui::TreeNode("UTF-8 Text"))
//		{
//			// UTF-8 test with Japanese characters
//			// (Needs a suitable font, try Noto, or Arial Unicode, or M+ fonts. Read misc/fonts/README.txt for details.)
//			// - From C++11 you can use the u8"my text" syntax to encode literal strings as UTF-8
//			// - For earlier compiler, you may be able to encode your sources as UTF-8 (e.g. Visual Studio save your file as 'UTF-8 without signature')
//			// - FOR THIS DEMO FILE ONLY, BECAUSE WE WANT TO SUPPORT OLD COMPILERS, WE ARE *NOT* INCLUDING RAW UTF-8 CHARACTERS IN THIS SOURCE FILE.
//			//   Instead we are encoding a few strings with hexadecimal constants. Don't do this in your application!
//			//   Please use u8"text in any language" in your application!
//			// Note that characters values are preserved even by InputText() if the font cannot be displayed, so you can safely copy & paste garbled characters into another application.
//			ImGui::TextWrapped("CJK text will only appears if the font was loaded with the appropriate CJK character ranges. Call io.Font->AddFontFromFileTTF() manually to load extra character ranges. Read misc/fonts/README.txt for details.");
//			ImGui::Text("Hiragana: \xe3\x81\x8b\xe3\x81\x8d\xe3\x81\x8f\xe3\x81\x91\xe3\x81\x93 (kakikukeko)"); // Normally we would use u8"blah blah" with the proper characters directly in the string.
//			ImGui::Text("Kanjis: \xe6\x97\xa5\xe6\x9c\xac\xe8\xaa\x9e (nihongo)");
//			static char buf[32] = "\xe6\x97\xa5\xe6\x9c\xac\xe8\xaa\x9e";
//			//static char buf[32] = u8"NIHONGO"; // <- this is how you would write it with C++11, using real kanjis
//			ImGui::InputText("UTF-8 input", buf, IM_ARRAYSIZE(buf));
//			ImGui::TreePop();
//		}
//		ImGui::TreePop();
//	}
//
//	if (ImGui::TreeNode("Images"))
//	{
//		ImGuiIO& io = ImGui::GetIO();
//		ImGui::TextWrapped("Below we are displaying the font texture (which is the only texture we have access to in this demo). Use the 'ImTextureID' type as storage to pass pointers or identifier to your own texture data. Hover the texture for a zoomed view!");
//
//		// Here we are grabbing the font texture because that's the only one we have access to inside the demo code.
//		// Remember that ImTextureID is just storage for whatever you want it to be, it is essentially a value that will be passed to the render function inside the ImDrawCmd structure.
//		// If you use one of the default imgui_impl_XXXX.cpp renderer, they all have comments at the top of their file to specify what they expect to be stored in ImTextureID.
//		// (for example, the imgui_impl_dx11.cpp renderer expect a 'ID3D11ShaderResourceView*' pointer. The imgui_impl_glfw_gl3.cpp renderer expect a GLuint OpenGL texture identifier etc.)
//		// If you decided that ImTextureID = MyEngineTexture*, then you can pass your MyEngineTexture* pointers to ImGui::Image(), and gather width/height through your own functions, etc.
//		// Using ShowMetricsWindow() as a "debugger" to inspect the draw data that are being passed to your render will help you debug issues if you are confused about this.
//		// Consider using the lower-level ImDrawList::AddImage() API, via ImGui::GetWindowDrawList()->AddImage().
//		ImTextureID my_tex_id = io.Fonts->TexID;
//		float my_tex_w = (float)io.Fonts->TexWidth;
//		float my_tex_h = (float)io.Fonts->TexHeight;
//
//		ImGui::Text("%.0fx%.0f", my_tex_w, my_tex_h);
//		ImVec2 pos = ImGui::GetCursorScreenPos();
//		ImGui::Image(my_tex_id, ImVec2(my_tex_w, my_tex_h), ImVec2(0, 0), ImVec2(1, 1), ImVec4(1.0f, 1.0f, 1.0f, 1.0f), ImVec4(1.0f, 1.0f, 1.0f, 0.5f));
//		if (ImGui::IsItemHovered())
//		{
//			ImGui::BeginTooltip();
//			float region_sz = 32.0f;
//			float region_x = io.MousePos.x - pos.x - region_sz * 0.5f; if (region_x < 0.0f) region_x = 0.0f; else if (region_x > my_tex_w - region_sz) region_x = my_tex_w - region_sz;
//			float region_y = io.MousePos.y - pos.y - region_sz * 0.5f; if (region_y < 0.0f) region_y = 0.0f; else if (region_y > my_tex_h - region_sz) region_y = my_tex_h - region_sz;
//			float zoom = 4.0f;
//			ImGui::Text("Min: (%.2f, %.2f)", region_x, region_y);
//			ImGui::Text("Max: (%.2f, %.2f)", region_x + region_sz, region_y + region_sz);
//			ImVec2 uv0 = ImVec2((region_x) / my_tex_w, (region_y) / my_tex_h);
//			ImVec2 uv1 = ImVec2((region_x + region_sz) / my_tex_w, (region_y + region_sz) / my_tex_h);
//			ImGui::Image(my_tex_id, ImVec2(region_sz * zoom, region_sz * zoom), uv0, uv1, ImVec4(1.0f, 1.0f, 1.0f, 1.0f), ImVec4(1.0f, 1.0f, 1.0f, 0.5f));
//			ImGui::EndTooltip();
//		}
//		ImGui::TextWrapped("And now some textured buttons..");
//		static int pressed_count = 0;
//		for (int i = 0; i < 8; i++)
//		{
//			ImGui::PushID(i);
//			int frame_padding = -1 + i;     // -1 = uses default padding
//			if (ImGui::ImageButton(my_tex_id, ImVec2(32, 32), ImVec2(0, 0), ImVec2(32.0f / my_tex_w, 32 / my_tex_h), frame_padding, ImVec4(0.0f, 0.0f, 0.0f, 1.0f)))
//				pressed_count += 1;
//			ImGui::PopID();
//			ImGui::SameLine();
//		}
//		ImGui::NewLine();
//		ImGui::Text("Pressed %d times.", pressed_count);
//		ImGui::TreePop();
//	}
//
//	if (ImGui::TreeNode("Combo"))
//	{
//		// Expose flags as checkbox for the demo
//		static ImGuiComboFlags flags = 0;
//		ImGui::CheckboxFlags("ImGuiComboFlags_PopupAlignLeft", (unsigned int*)&flags, ImGuiComboFlags_PopupAlignLeft);
//		ImGui::SameLine(); HelpMarker("Only makes a difference if the popup is larger than the combo");
//		if (ImGui::CheckboxFlags("ImGuiComboFlags_NoArrowButton", (unsigned int*)&flags, ImGuiComboFlags_NoArrowButton))
//			flags &= ~ImGuiComboFlags_NoPreview;     // Clear the other flag, as we cannot combine both
//		if (ImGui::CheckboxFlags("ImGuiComboFlags_NoPreview", (unsigned int*)&flags, ImGuiComboFlags_NoPreview))
//			flags &= ~ImGuiComboFlags_NoArrowButton; // Clear the other flag, as we cannot combine both
//
//		// General BeginCombo() API, you have full control over your selection data and display type.
//		// (your selection data could be an index, a pointer to the object, an id for the object, a flag stored in the object itself, etc.)
//		const char* items[] = { "AAAA", "BBBB", "CCCC", "DDDD", "EEEE", "FFFF", "GGGG", "HHHH", "IIII", "JJJJ", "KKKK", "LLLLLLL", "MMMM", "OOOOOOO" };
//		static const char* item_current = items[0];            // Here our selection is a single pointer stored outside the object.
//		if (ImGui::BeginCombo("combo 1", item_current, flags)) // The second parameter is the label previewed before opening the combo.
//		{
//			for (int n = 0; n < IM_ARRAYSIZE(items); n++)
//			{
//				bool is_selected = (item_current == items[n]);
//				if (ImGui::Selectable(items[n], is_selected))
//					item_current = items[n];
//				if (is_selected)
//					ImGui::SetItemDefaultFocus();   // Set the initial focus when opening the combo (scrolling + for keyboard navigation support in the upcoming navigation branch)
//			}
//			ImGui::EndCombo();
//		}
//
//		// Simplified one-liner Combo() API, using values packed in a single constant string
//		static int item_current_2 = 0;
//		ImGui::Combo("combo 2 (one-liner)", &item_current_2, "aaaa\0bbbb\0cccc\0dddd\0eeee\0\0");
//
//		// Simplified one-liner Combo() using an array of const char*
//		static int item_current_3 = -1; // If the selection isn't within 0..count, Combo won't display a preview
//		ImGui::Combo("combo 3 (array)", &item_current_3, items, IM_ARRAYSIZE(items));
//
//		// Simplified one-liner Combo() using an accessor function
//		struct FuncHolder { static bool ItemGetter(void* data, int idx, const char** out_str) { *out_str = ((const char**)data)[idx]; return true; } };
//		static int item_current_4 = 0;
//		ImGui::Combo("combo 4 (function)", &item_current_4, &FuncHolder::ItemGetter, items, IM_ARRAYSIZE(items));
//
//		ImGui::TreePop();
//	}
//
//	if (ImGui::TreeNode("Selectables"))
//	{
//		// Selectable() has 2 overloads:
//		// - The one taking "bool selected" as a read-only selection information. When Selectable() has been clicked is returns true and you can alter selection state accordingly.
//		// - The one taking "bool* p_selected" as a read-write selection information (convenient in some cases)
//		// The earlier is more flexible, as in real application your selection may be stored in a different manner (in flags within objects, as an external list, etc).
//		if (ImGui::TreeNode("Basic"))
//		{
//			static bool selection[5] = { false, true, false, false, false };
//			ImGui::Selectable("1. I am selectable", &selection[0]);
//			ImGui::Selectable("2. I am selectable", &selection[1]);
//			ImGui::Text("3. I am not selectable");
//			ImGui::Selectable("4. I am selectable", &selection[3]);
//			if (ImGui::Selectable("5. I am double clickable", selection[4], ImGuiSelectableFlags_AllowDoubleClick))
//				if (ImGui::IsMouseDoubleClicked(0))
//					selection[4] = !selection[4];
//			ImGui::TreePop();
//		}
//		if (ImGui::TreeNode("Selection State: Single Selection"))
//		{
//			static int selected = -1;
//			for (int n = 0; n < 5; n++)
//			{
//				char buf[32];
//				sprintf(buf, "Object %d", n);
//				if (ImGui::Selectable(buf, selected == n))
//					selected = n;
//			}
//			ImGui::TreePop();
//		}
//		if (ImGui::TreeNode("Selection State: Multiple Selection"))
//		{
//			HelpMarker("Hold CTRL and click to select multiple items.");
//			static bool selection[5] = { false, false, false, false, false };
//			for (int n = 0; n < 5; n++)
//			{
//				char buf[32];
//				sprintf(buf, "Object %d", n);
//				if (ImGui::Selectable(buf, selection[n]))
//				{
//					if (!ImGui::GetIO().KeyCtrl)    // Clear selection when CTRL is not held
//						memset(selection, 0, sizeof(selection));
//					selection[n] ^= 1;
//				}
//			}
//			ImGui::TreePop();
//		}
//		if (ImGui::TreeNode("Rendering more text into the same line"))
//		{
//			// Using the Selectable() override that takes "bool* p_selected" parameter and toggle your booleans automatically.
//			static bool selected[3] = { false, false, false };
//			ImGui::Selectable("main.c", &selected[0]); ImGui::SameLine(300); ImGui::Text(" 2,345 bytes");
//			ImGui::Selectable("Hello.cpp", &selected[1]); ImGui::SameLine(300); ImGui::Text("12,345 bytes");
//			ImGui::Selectable("Hello.h", &selected[2]); ImGui::SameLine(300); ImGui::Text(" 2,345 bytes");
//			ImGui::TreePop();
//		}
//		if (ImGui::TreeNode("In columns"))
//		{
//			ImGui::Columns(3, NULL, false);
//			static bool selected[16] = { 0 };
//			for (int i = 0; i < 16; i++)
//			{
//				char label[32]; sprintf(label, "Item %d", i);
//				if (ImGui::Selectable(label, &selected[i])) {}
//				ImGui::NextColumn();
//			}
//			ImGui::Columns(1);
//			ImGui::TreePop();
//		}
//		if (ImGui::TreeNode("Grid"))
//		{
//			static bool selected[4 * 4] = { true, false, false, false, false, true, false, false, false, false, true, false, false, false, false, true };
//			for (int i = 0; i < 4 * 4; i++)
//			{
//				ImGui::PushID(i);
//				if (ImGui::Selectable("Sailor", &selected[i], 0, ImVec2(50, 50)))
//				{
//					// Note: We _unnecessarily_ test for both x/y and i here only to silence some static analyzer. The second part of each test is unnecessary.
//					int x = i % 4;
//					int y = i / 4;
//					if (x > 0)           { selected[i - 1] ^= 1; }
//					if (x < 3 && i < 15) { selected[i + 1] ^= 1; }
//					if (y > 0 && i > 3)  { selected[i - 4] ^= 1; }
//					if (y < 3 && i < 12) { selected[i + 4] ^= 1; }
//				}
//				if ((i % 4) < 3) ImGui::SameLine();
//				ImGui::PopID();
//			}
//			ImGui::TreePop();
//		}
//		if (ImGui::TreeNode("Alignment"))
//		{
//			HelpMarker("Alignment applies when a selectable is larger than its text content.\nBy default, Selectables uses style.SelectableTextAlign but it can be overriden on a per-item basis using PushStyleVar().");
//			static bool selected[3 * 3] = { true, false, true, false, true, false, true, false, true };
//			for (int y = 0; y < 3; y++)
//			{
//				for (int x = 0; x < 3; x++)
//				{
//					ImVec2 alignment = ImVec2((float)x / 2.0f, (float)y / 2.0f);
//					char name[32];
//					sprintf(name, "(%.1f,%.1f)", alignment.x, alignment.y);
//					if (x > 0) ImGui::SameLine();
//					ImGui::PushStyleVar(ImGuiStyleVar_SelectableTextAlign, alignment);
//					ImGui::Selectable(name, &selected[3 * y + x], ImGuiSelectableFlags_None, ImVec2(80, 80));
//					ImGui::PopStyleVar();
//				}
//			}
//			ImGui::TreePop();
//		}
//		ImGui::TreePop();
//	}
//
//	if (ImGui::TreeNode("Text Input"))
//	{
//		if (ImGui::TreeNode("Multi-line Text Input"))
//		{
//			// Note: we are using a fixed-sized buffer for simplicity here. See ImGuiInputTextFlags_CallbackResize
//			// and the code in misc/cpp/imgui_stdlib.h for how to setup InputText() for dynamically resizing strings.
//			static char text[1024 * 16] =
//				"/*\n"
//				" The Pentium F00F bug, shorthand for F0 0F C7 C8,\n"
//				" the hexadecimal encoding of one offending instruction,\n"
//				" more formally, the invalid operand with locked CMPXCHG8B\n"
//				" instruction bug, is a design flaw in the majority of\n"
//				" Intel Pentium, Pentium MMX, and Pentium OverDrive\n"
//				" processors (all in the P5 microarchitecture).\n"
//				"*/\n\n"
//				"label:\n"
//				"\tlock cmpxchg8b eax\n";
//
//			static ImGuiInputTextFlags flags = ImGuiInputTextFlags_AllowTabInput;
//			HelpMarker("You can use the ImGuiInputTextFlags_CallbackResize facility if you need to wire InputTextMultiline() to a dynamic string type. See misc/cpp/imgui_stdlib.h for an example. (This is not demonstrated in imgui_demo.cpp)");
//			ImGui::CheckboxFlags("ImGuiInputTextFlags_ReadOnly", (unsigned int*)&flags, ImGuiInputTextFlags_ReadOnly);
//			ImGui::CheckboxFlags("ImGuiInputTextFlags_AllowTabInput", (unsigned int*)&flags, ImGuiInputTextFlags_AllowTabInput);
//			ImGui::CheckboxFlags("ImGuiInputTextFlags_CtrlEnterForNewLine", (unsigned int*)&flags, ImGuiInputTextFlags_CtrlEnterForNewLine);
//			ImGui::InputTextMultiline("##source", text, IM_ARRAYSIZE(text), ImVec2(-FLT_MIN, ImGui::GetTextLineHeight() * 16), flags);
//			ImGui::TreePop();
//		}
//
//		if (ImGui::TreeNode("Filtered Text Input"))
//		{
//			static char buf1[64] = ""; ImGui::InputText("default", buf1, 64);
//			static char buf2[64] = ""; ImGui::InputText("decimal", buf2, 64, ImGuiInputTextFlags_CharsDecimal);
//			static char buf3[64] = ""; ImGui::InputText("hexadecimal", buf3, 64, ImGuiInputTextFlags_CharsHexadecimal | ImGuiInputTextFlags_CharsUppercase);
//			static char buf4[64] = ""; ImGui::InputText("uppercase", buf4, 64, ImGuiInputTextFlags_CharsUppercase);
//			static char buf5[64] = ""; ImGui::InputText("no blank", buf5, 64, ImGuiInputTextFlags_CharsNoBlank);
//			struct TextFilters { static int FilterImGuiLetters(ImGuiInputTextCallbackData* data) { if (data->EventChar < 256 && strchr("imgui", (char)data->EventChar)) return 0; return 1; } };
//			static char buf6[64] = ""; ImGui::InputText("\"imgui\" letters", buf6, 64, ImGuiInputTextFlags_CallbackCharFilter, TextFilters::FilterImGuiLetters);
//
//			ImGui::Text("Password input");
//			static char bufpass[64] = "password123";
//			ImGui::InputText("password", bufpass, 64, ImGuiInputTextFlags_Password | ImGuiInputTextFlags_CharsNoBlank);
//			ImGui::SameLine(); HelpMarker("Display all characters as '*'.\nDisable clipboard cut and copy.\nDisable logging.\n");
//			ImGui::InputTextWithHint("password (w/ hint)", "<password>", bufpass, 64, ImGuiInputTextFlags_Password | ImGuiInputTextFlags_CharsNoBlank);
//			ImGui::InputText("password (clear)", bufpass, 64, ImGuiInputTextFlags_CharsNoBlank);
//			ImGui::TreePop();
//		}
//
//		if (ImGui::TreeNode("Resize Callback"))
//		{
//			// If you have a custom string type you would typically create a ImGui::InputText() wrapper than takes your type as input.
//			// See misc/cpp/imgui_stdlib.h and .cpp for an implementation of this using std::string.
//			HelpMarker("Demonstrate using ImGuiInputTextFlags_CallbackResize to wire your resizable string type to InputText().\n\nSee misc/cpp/imgui_stdlib.h for an implementation of this for std::string.");
//			struct Funcs
//			{
//				static int MyResizeCallback(ImGuiInputTextCallbackData* data)
//				{
//					if (data->EventFlag == ImGuiInputTextFlags_CallbackResize)
//					{
//						ImVector<char>* my_str = (ImVector<char>*)data->UserData;
//						IM_ASSERT(my_str->begin() == data->Buf);
//						my_str->resize(data->BufSize);  // NB: On resizing calls, generally data->BufSize == data->BufTextLen + 1
//						data->Buf = my_str->begin();
//					}
//					return 0;
//				}
//
//				// Tip: Because ImGui:: is a namespace you would typicall add your own function into the namespace in your own source files.
//				// For example, you may add a function called ImGui::InputText(const char* label, MyString* my_str).
//				static bool MyInputTextMultiline(const char* label, ImVector<char>* my_str, const ImVec2& size = ImVec2(0, 0), ImGuiInputTextFlags flags = 0)
//				{
//					IM_ASSERT((flags & ImGuiInputTextFlags_CallbackResize) == 0);
//					return ImGui::InputTextMultiline(label, my_str->begin(), (size_t)my_str->size(), size, flags | ImGuiInputTextFlags_CallbackResize, Funcs::MyResizeCallback, (void*)my_str);
//				}
//			};
//
//			// For this demo we are using ImVector as a string container.
//			// Note that because we need to store a terminating zero character, our size/capacity are 1 more than usually reported by a typical string class.
//			static ImVector<char> my_str;
//			if (my_str.empty())
//				my_str.push_back(0);
//			Funcs::MyInputTextMultiline("##MyStr", &my_str, ImVec2(-FLT_MIN, ImGui::GetTextLineHeight() * 16));
//			ImGui::Text("Data: %p\nSize: %d\nCapacity: %d", (void*)my_str.begin(), my_str.size(), my_str.capacity());
//			ImGui::TreePop();
//		}
//
//		ImGui::TreePop();
//	}
//
//	if (ImGui::TreeNode("Plots Widgets"))
//	{
//		static bool animate = true;
//		ImGui::Checkbox("Animate", &animate);
//
//		static float arr[] = { 0.6f, 0.1f, 1.0f, 0.5f, 0.92f, 0.1f, 0.2f };
//		ImGui::PlotLines("Frame Times", arr, IM_ARRAYSIZE(arr));
//
//		// Create a dummy array of contiguous float values to plot
//		// Tip: If your float aren't contiguous but part of a structure, you can pass a pointer to your first float and the sizeof() of your structure in the Stride parameter.
//		static float values[90] = { 0 };
//		static int values_offset = 0;
//		static double refresh_time = 0.0;
//		if (!animate || refresh_time == 0.0)
//			refresh_time = ImGui::GetTime();
//		while (refresh_time < ImGui::GetTime()) // Create dummy data at fixed 60 hz rate for the demo
//		{
//			static float phase = 0.0f;
//			values[values_offset] = cosf(phase);
//			values_offset = (values_offset + 1) % IM_ARRAYSIZE(values);
//			phase += 0.10f*values_offset;
//			refresh_time += 1.0f / 60.0f;
//		}
//		ImGui::PlotLines("Lines", values, IM_ARRAYSIZE(values), values_offset, "avg 0.0", -1.0f, 1.0f, ImVec2(0, 80));
//		ImGui::PlotHistogram("Histogram", arr, IM_ARRAYSIZE(arr), 0, NULL, 0.0f, 1.0f, ImVec2(0, 80));
//
//		// Use functions to generate output
//		// FIXME: This is rather awkward because current plot API only pass in indices. We probably want an API passing floats and user provide sample rate/count.
//		struct Funcs
//		{
//			static float Sin(void*, int i) { return sinf(i * 0.1f); }
//			static float Saw(void*, int i) { return (i & 1) ? 1.0f : -1.0f; }
//		};
//		static int func_type = 0, display_count = 70;
//		ImGui::Separator();
//		ImGui::SetNextItemWidth(100);
//		ImGui::Combo("func", &func_type, "Sin\0Saw\0");
//		ImGui::SameLine();
//		ImGui::SliderInt("Sample count", &display_count, 1, 400);
//		float(*func)(void*, int) = (func_type == 0) ? Funcs::Sin : Funcs::Saw;
//		ImGui::PlotLines("Lines", func, NULL, display_count, 0, NULL, -1.0f, 1.0f, ImVec2(0, 80));
//		ImGui::PlotHistogram("Histogram", func, NULL, display_count, 0, NULL, -1.0f, 1.0f, ImVec2(0, 80));
//		ImGui::Separator();
//
//		// Animate a simple progress bar
//		static float progress = 0.0f, progress_dir = 1.0f;
//		if (animate)
//		{
//			progress += progress_dir * 0.4f * ImGui::GetIO().DeltaTime;
//			if (progress >= +1.1f) { progress = +1.1f; progress_dir *= -1.0f; }
//			if (progress <= -0.1f) { progress = -0.1f; progress_dir *= -1.0f; }
//		}
//
//		// Typically we would use ImVec2(-1.0f,0.0f) or ImVec2(-FLT_MIN,0.0f) to use all available width, 
//		// or ImVec2(width,0.0f) for a specified width. ImVec2(0.0f,0.0f) uses ItemWidth.
//		ImGui::ProgressBar(progress, ImVec2(0.0f, 0.0f));
//		ImGui::SameLine(0.0f, ImGui::GetStyle().ItemInnerSpacing.x);
//		ImGui::Text("Progress Bar");
//
//		float progress_saturated = (progress < 0.0f) ? 0.0f : (progress > 1.0f) ? 1.0f : progress;
//		char buf[32];
//		sprintf(buf, "%d/%d", (int)(progress_saturated * 1753), 1753);
//		ImGui::ProgressBar(progress, ImVec2(0.f, 0.f), buf);
//		ImGui::TreePop();
//	}
//
//	if (ImGui::TreeNode("Color/Picker Widgets"))
//	{
//		static ImVec4 color = ImVec4(114.0f / 255.0f, 144.0f / 255.0f, 154.0f / 255.0f, 200.0f / 255.0f);
//
//		static bool alpha_preview = true;
//		static bool alpha_half_preview = false;
//		static bool drag_and_drop = true;
//		static bool options_menu = true;
//		static bool hdr = false;
//		ImGui::Checkbox("With Alpha Preview", &alpha_preview);
//		ImGui::Checkbox("With Half Alpha Preview", &alpha_half_preview);
//		ImGui::Checkbox("With Drag and Drop", &drag_and_drop);
//		ImGui::Checkbox("With Options Menu", &options_menu); ImGui::SameLine(); HelpMarker("Right-click on the individual color widget to show options.");
//		ImGui::Checkbox("With HDR", &hdr); ImGui::SameLine(); HelpMarker("Currently all this does is to lift the 0..1 limits on dragging widgets.");
//		int misc_flags = (hdr ? ImGuiColorEditFlags_HDR : 0) | (drag_and_drop ? 0 : ImGuiColorEditFlags_NoDragDrop) | (alpha_half_preview ? ImGuiColorEditFlags_AlphaPreviewHalf : (alpha_preview ? ImGuiColorEditFlags_AlphaPreview : 0)) | (options_menu ? 0 : ImGuiColorEditFlags_NoOptions);
//
//		ImGui::Text("Color widget:");
//		ImGui::SameLine(); HelpMarker("Click on the colored square to open a color picker.\nCTRL+click on individual component to input value.\n");
//		ImGui::ColorEdit3("MyColor##1", (float*)&color, misc_flags);
//
//		ImGui::Text("Color widget HSV with Alpha:");
//		ImGui::ColorEdit4("MyColor##2", (float*)&color, ImGuiColorEditFlags_DisplayHSV | misc_flags);
//
//		ImGui::Text("Color widget with Float Display:");
//		ImGui::ColorEdit4("MyColor##2f", (float*)&color, ImGuiColorEditFlags_Float | misc_flags);
//
//		ImGui::Text("Color button with Picker:");
//		ImGui::SameLine(); HelpMarker("With the ImGuiColorEditFlags_NoInputs flag you can hide all the slider/text inputs.\nWith the ImGuiColorEditFlags_NoLabel flag you can pass a non-empty label which will only be used for the tooltip and picker popup.");
//		ImGui::ColorEdit4("MyColor##3", (float*)&color, ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_NoLabel | misc_flags);
//
//		ImGui::Text("Color button with Custom Picker Popup:");
//
//		// Generate a dummy default palette. The palette will persist and can be edited.
//		static bool saved_palette_init = true;
//		static ImVec4 saved_palette[32] = {};
//		if (saved_palette_init)
//		{
//			for (int n = 0; n < IM_ARRAYSIZE(saved_palette); n++)
//			{
//				ImGui::ColorConvertHSVtoRGB(n / 31.0f, 0.8f, 0.8f, saved_palette[n].x, saved_palette[n].y, saved_palette[n].z);
//				saved_palette[n].w = 1.0f; // Alpha
//			}
//			saved_palette_init = false;
//		}
//
//		static ImVec4 backup_color;
//		bool open_popup = ImGui::ColorButton("MyColor##3b", color, misc_flags);
//		ImGui::SameLine();
//		open_popup |= ImGui::Button("Palette");
//		if (open_popup)
//		{
//			ImGui::OpenPopup("mypicker");
//			backup_color = color;
//		}
//		if (ImGui::BeginPopup("mypicker"))
//		{
//			ImGui::Text("MY CUSTOM COLOR PICKER WITH AN AMAZING PALETTE!");
//			ImGui::Separator();
//			ImGui::ColorPicker4("##picker", (float*)&color, misc_flags | ImGuiColorEditFlags_NoSidePreview | ImGuiColorEditFlags_NoSmallPreview);
//			ImGui::SameLine();
//
//			ImGui::BeginGroup(); // Lock X position
//			ImGui::Text("Current");
//			ImGui::ColorButton("##current", color, ImGuiColorEditFlags_NoPicker | ImGuiColorEditFlags_AlphaPreviewHalf, ImVec2(60, 40));
//			ImGui::Text("Previous");
//			if (ImGui::ColorButton("##previous", backup_color, ImGuiColorEditFlags_NoPicker | ImGuiColorEditFlags_AlphaPreviewHalf, ImVec2(60, 40)))
//				color = backup_color;
//			ImGui::Separator();
//			ImGui::Text("Palette");
//			for (int n = 0; n < IM_ARRAYSIZE(saved_palette); n++)
//			{
//				ImGui::PushID(n);
//				if ((n % 8) != 0)
//					ImGui::SameLine(0.0f, ImGui::GetStyle().ItemSpacing.y);
//				if (ImGui::ColorButton("##palette", saved_palette[n], ImGuiColorEditFlags_NoAlpha | ImGuiColorEditFlags_NoPicker | ImGuiColorEditFlags_NoTooltip, ImVec2(20, 20)))
//					color = ImVec4(saved_palette[n].x, saved_palette[n].y, saved_palette[n].z, color.w); // Preserve alpha!
//
//				// Allow user to drop colors into each palette entry
//				// (Note that ColorButton is already a drag source by default, unless using ImGuiColorEditFlags_NoDragDrop)
//				if (ImGui::BeginDragDropTarget())
//				{
//					if (const ImGuiPayload* payload = ImGui::AcceptDragDropPayload(IMGUI_PAYLOAD_TYPE_COLOR_3F))
//						memcpy((float*)&saved_palette[n], payload->Data, sizeof(float) * 3);
//					if (const ImGuiPayload* payload = ImGui::AcceptDragDropPayload(IMGUI_PAYLOAD_TYPE_COLOR_4F))
//						memcpy((float*)&saved_palette[n], payload->Data, sizeof(float) * 4);
//					ImGui::EndDragDropTarget();
//				}
//
//				ImGui::PopID();
//			}
//			ImGui::EndGroup();
//			ImGui::EndPopup();
//		}
//
//		ImGui::Text("Color button only:");
//		ImGui::ColorButton("MyColor##3c", *(ImVec4*)&color, misc_flags, ImVec2(80, 80));
//
//		ImGui::Text("Color picker:");
//		static bool alpha = true;
//		static bool alpha_bar = true;
//		static bool side_preview = true;
//		static bool ref_color = false;
//		static ImVec4 ref_color_v(1.0f, 0.0f, 1.0f, 0.5f);
//		static int display_mode = 0;
//		static int picker_mode = 0;
//		ImGui::Checkbox("With Alpha", &alpha);
//		ImGui::Checkbox("With Alpha Bar", &alpha_bar);
//		ImGui::Checkbox("With Side Preview", &side_preview);
//		if (side_preview)
//		{
//			ImGui::SameLine();
//			ImGui::Checkbox("With Ref Color", &ref_color);
//			if (ref_color)
//			{
//				ImGui::SameLine();
//				ImGui::ColorEdit4("##RefColor", &ref_color_v.x, ImGuiColorEditFlags_NoInputs | misc_flags);
//			}
//		}
//		ImGui::Combo("Display Mode", &display_mode, "Auto/Current\0None\0RGB Only\0HSV Only\0Hex Only\0");
//		ImGui::SameLine(); HelpMarker("ColorEdit defaults to displaying RGB inputs if you don't specify a display mode, but the user can change it with a right-click.\n\nColorPicker defaults to displaying RGB+HSV+Hex if you don't specify a display mode.\n\nYou can change the defaults using SetColorEditOptions().");
//		ImGui::Combo("Picker Mode", &picker_mode, "Auto/Current\0Hue bar + SV rect\0Hue wheel + SV triangle\0");
//		ImGui::SameLine(); HelpMarker("User can right-click the picker to change mode.");
//		ImGuiColorEditFlags flags = misc_flags;
//		if (!alpha)            flags |= ImGuiColorEditFlags_NoAlpha;        // This is by default if you call ColorPicker3() instead of ColorPicker4()
//		if (alpha_bar)         flags |= ImGuiColorEditFlags_AlphaBar;
//		if (!side_preview)     flags |= ImGuiColorEditFlags_NoSidePreview;
//		if (picker_mode == 1)  flags |= ImGuiColorEditFlags_PickerHueBar;
//		if (picker_mode == 2)  flags |= ImGuiColorEditFlags_PickerHueWheel;
//		if (display_mode == 1) flags |= ImGuiColorEditFlags_NoInputs;       // Disable all RGB/HSV/Hex displays
//		if (display_mode == 2) flags |= ImGuiColorEditFlags_DisplayRGB;     // Override display mode
//		if (display_mode == 3) flags |= ImGuiColorEditFlags_DisplayHSV;
//		if (display_mode == 4) flags |= ImGuiColorEditFlags_DisplayHex;
//		ImGui::ColorPicker4("MyColor##4", (float*)&color, flags, ref_color ? &ref_color_v.x : NULL);
//
//		ImGui::Text("Programmatically set defaults:");
//		ImGui::SameLine(); HelpMarker("SetColorEditOptions() is designed to allow you to set boot-time default.\nWe don't have Push/Pop functions because you can force options on a per-widget basis if needed, and the user can change non-forced ones with the options menu.\nWe don't have a getter to avoid encouraging you to persistently save values that aren't forward-compatible.");
//		if (ImGui::Button("Default: Uint8 + HSV + Hue Bar"))
//			ImGui::SetColorEditOptions(ImGuiColorEditFlags_Uint8 | ImGuiColorEditFlags_DisplayHSV | ImGuiColorEditFlags_PickerHueBar);
//		if (ImGui::Button("Default: Float + HDR + Hue Wheel"))
//			ImGui::SetColorEditOptions(ImGuiColorEditFlags_Float | ImGuiColorEditFlags_HDR | ImGuiColorEditFlags_PickerHueWheel);
//
//		// HSV encoded support (to avoid RGB<>HSV round trips and singularities when S==0 or V==0)
//		static ImVec4 color_stored_as_hsv(0.23f, 1.0f, 1.0f, 1.0f);
//		ImGui::Spacing();
//		ImGui::Text("HSV encoded colors");
//		ImGui::SameLine(); HelpMarker("By default, colors are given to ColorEdit and ColorPicker in RGB, but ImGuiColorEditFlags_InputHSV allows you to store colors as HSV and pass them to ColorEdit and ColorPicker as HSV. This comes with the added benefit that you can manipulate hue values with the picker even when saturation or value are zero.");
//		ImGui::Text("Color widget with InputHSV:");
//		ImGui::ColorEdit4("HSV shown as RGB##1", (float*)&color_stored_as_hsv, ImGuiColorEditFlags_DisplayRGB | ImGuiColorEditFlags_InputHSV | ImGuiColorEditFlags_Float);
//		ImGui::ColorEdit4("HSV shown as HSV##1", (float*)&color_stored_as_hsv, ImGuiColorEditFlags_DisplayHSV | ImGuiColorEditFlags_InputHSV | ImGuiColorEditFlags_Float);
//		ImGui::DragFloat4("Raw HSV values", (float*)&color_stored_as_hsv, 0.01f, 0.0f, 1.0f);
//
//		ImGui::TreePop();
//	}
//
//	if (ImGui::TreeNode("Range Widgets"))
//	{
//		static float begin = 10, end = 90;
//		static int begin_i = 100, end_i = 1000;
//		ImGui::DragFloatRange2("range", &begin, &end, 0.25f, 0.0f, 100.0f, "Min: %.1f %%", "Max: %.1f %%");
//		ImGui::DragIntRange2("range int (no bounds)", &begin_i, &end_i, 5, 0, 0, "Min: %d units", "Max: %d units");
//		ImGui::TreePop();
//	}
//
//	if (ImGui::TreeNode("Data Types"))
//	{
//		// The DragScalar/InputScalar/SliderScalar functions allow various data types: signed/unsigned int/long long and float/double
//		// To avoid polluting the public API with all possible combinations, we use the ImGuiDataType enum to pass the type,
//		// and passing all arguments by address.
//		// This is the reason the test code below creates local variables to hold "zero" "one" etc. for each types.
//		// In practice, if you frequently use a given type that is not covered by the normal API entry points, you can wrap it
//		// yourself inside a 1 line function which can take typed argument as value instead of void*, and then pass their address
//		// to the generic function. For example:
//		//   bool MySliderU64(const char *label, u64* value, u64 min = 0, u64 max = 0, const char* format = "%lld")
//		//   {
//		//      return SliderScalar(label, ImGuiDataType_U64, value, &min, &max, format);
//		//   }
//
//		// Limits (as helper variables that we can take the address of)
//		// Note that the SliderScalar function has a maximum usable range of half the natural type maximum, hence the /2 below.
//#ifndef LLONG_MIN
//		ImS64 LLONG_MIN = -9223372036854775807LL - 1;
//		ImS64 LLONG_MAX = 9223372036854775807LL;
//		ImU64 ULLONG_MAX = (2ULL * 9223372036854775807LL + 1);
//#endif
//		const char    s8_zero = 0, s8_one = 1, s8_fifty = 50, s8_min = -128, s8_max = 127;
//		const ImU8    u8_zero = 0, u8_one = 1, u8_fifty = 50, u8_min = 0, u8_max = 255;
//		const short   s16_zero = 0, s16_one = 1, s16_fifty = 50, s16_min = -32768, s16_max = 32767;
//		const ImU16   u16_zero = 0, u16_one = 1, u16_fifty = 50, u16_min = 0, u16_max = 65535;
//		const ImS32   s32_zero = 0, s32_one = 1, s32_fifty = 50, s32_min = INT_MIN / 2, s32_max = INT_MAX / 2, s32_hi_a = INT_MAX / 2 - 100, s32_hi_b = INT_MAX / 2;
//		const ImU32   u32_zero = 0, u32_one = 1, u32_fifty = 50, u32_min = 0, u32_max = UINT_MAX / 2, u32_hi_a = UINT_MAX / 2 - 100, u32_hi_b = UINT_MAX / 2;
//		const ImS64   s64_zero = 0, s64_one = 1, s64_fifty = 50, s64_min = LLONG_MIN / 2, s64_max = LLONG_MAX / 2, s64_hi_a = LLONG_MAX / 2 - 100, s64_hi_b = LLONG_MAX / 2;
//		const ImU64   u64_zero = 0, u64_one = 1, u64_fifty = 50, u64_min = 0, u64_max = ULLONG_MAX / 2, u64_hi_a = ULLONG_MAX / 2 - 100, u64_hi_b = ULLONG_MAX / 2;
//		const float   f32_zero = 0.f, f32_one = 1.f, f32_lo_a = -10000000000.0f, f32_hi_a = +10000000000.0f;
//		const double  f64_zero = 0., f64_one = 1., f64_lo_a = -1000000000000000.0, f64_hi_a = +1000000000000000.0;
//
//		// State
//		static char   s8_v = 127;
//		static ImU8   u8_v = 255;
//		static short  s16_v = 32767;
//		static ImU16  u16_v = 65535;
//		static ImS32  s32_v = -1;
//		static ImU32  u32_v = (ImU32)-1;
//		static ImS64  s64_v = -1;
//		static ImU64  u64_v = (ImU64)-1;
//		static float  f32_v = 0.123f;
//		static double f64_v = 90000.01234567890123456789;
//
//		const float drag_speed = 0.2f;
//		static bool drag_clamp = false;
//		ImGui::Text("Drags:");
//		ImGui::Checkbox("Clamp integers to 0..50", &drag_clamp); ImGui::SameLine(); HelpMarker("As with every widgets in dear imgui, we never modify values unless there is a user interaction.\nYou can override the clamping limits by using CTRL+Click to input a value.");
//		ImGui::DragScalar("drag s8", ImGuiDataType_S8, &s8_v, drag_speed, drag_clamp ? &s8_zero : NULL, drag_clamp ? &s8_fifty : NULL);
//		ImGui::DragScalar("drag u8", ImGuiDataType_U8, &u8_v, drag_speed, drag_clamp ? &u8_zero : NULL, drag_clamp ? &u8_fifty : NULL, "%u ms");
//		ImGui::DragScalar("drag s16", ImGuiDataType_S16, &s16_v, drag_speed, drag_clamp ? &s16_zero : NULL, drag_clamp ? &s16_fifty : NULL);
//		ImGui::DragScalar("drag u16", ImGuiDataType_U16, &u16_v, drag_speed, drag_clamp ? &u16_zero : NULL, drag_clamp ? &u16_fifty : NULL, "%u ms");
//		ImGui::DragScalar("drag s32", ImGuiDataType_S32, &s32_v, drag_speed, drag_clamp ? &s32_zero : NULL, drag_clamp ? &s32_fifty : NULL);
//		ImGui::DragScalar("drag u32", ImGuiDataType_U32, &u32_v, drag_speed, drag_clamp ? &u32_zero : NULL, drag_clamp ? &u32_fifty : NULL, "%u ms");
//		ImGui::DragScalar("drag s64", ImGuiDataType_S64, &s64_v, drag_speed, drag_clamp ? &s64_zero : NULL, drag_clamp ? &s64_fifty : NULL);
//		ImGui::DragScalar("drag u64", ImGuiDataType_U64, &u64_v, drag_speed, drag_clamp ? &u64_zero : NULL, drag_clamp ? &u64_fifty : NULL);
//		ImGui::DragScalar("drag float", ImGuiDataType_Float, &f32_v, 0.005f, &f32_zero, &f32_one, "%f", 1.0f);
//		ImGui::DragScalar("drag float ^2", ImGuiDataType_Float, &f32_v, 0.005f, &f32_zero, &f32_one, "%f", 2.0f); ImGui::SameLine(); HelpMarker("You can use the 'power' parameter to increase tweaking precision on one side of the range.");
//		ImGui::DragScalar("drag double", ImGuiDataType_Double, &f64_v, 0.0005f, &f64_zero, NULL, "%.10f grams", 1.0f);
//		ImGui::DragScalar("drag double ^2", ImGuiDataType_Double, &f64_v, 0.0005f, &f64_zero, &f64_one, "0 < %.10f < 1", 2.0f);
//
//		ImGui::Text("Sliders");
//		ImGui::SliderScalar("slider s8 full", ImGuiDataType_S8, &s8_v, &s8_min, &s8_max, "%d");
//		ImGui::SliderScalar("slider u8 full", ImGuiDataType_U8, &u8_v, &u8_min, &u8_max, "%u");
//		ImGui::SliderScalar("slider s16 full", ImGuiDataType_S16, &s16_v, &s16_min, &s16_max, "%d");
//		ImGui::SliderScalar("slider u16 full", ImGuiDataType_U16, &u16_v, &u16_min, &u16_max, "%u");
//		ImGui::SliderScalar("slider s32 low", ImGuiDataType_S32, &s32_v, &s32_zero, &s32_fifty, "%d");
//		ImGui::SliderScalar("slider s32 high", ImGuiDataType_S32, &s32_v, &s32_hi_a, &s32_hi_b, "%d");
//		ImGui::SliderScalar("slider s32 full", ImGuiDataType_S32, &s32_v, &s32_min, &s32_max, "%d");
//		ImGui::SliderScalar("slider u32 low", ImGuiDataType_U32, &u32_v, &u32_zero, &u32_fifty, "%u");
//		ImGui::SliderScalar("slider u32 high", ImGuiDataType_U32, &u32_v, &u32_hi_a, &u32_hi_b, "%u");
//		ImGui::SliderScalar("slider u32 full", ImGuiDataType_U32, &u32_v, &u32_min, &u32_max, "%u");
//		ImGui::SliderScalar("slider s64 low", ImGuiDataType_S64, &s64_v, &s64_zero, &s64_fifty, "%I64d");
//		ImGui::SliderScalar("slider s64 high", ImGuiDataType_S64, &s64_v, &s64_hi_a, &s64_hi_b, "%I64d");
//		ImGui::SliderScalar("slider s64 full", ImGuiDataType_S64, &s64_v, &s64_min, &s64_max, "%I64d");
//		ImGui::SliderScalar("slider u64 low", ImGuiDataType_U64, &u64_v, &u64_zero, &u64_fifty, "%I64u ms");
//		ImGui::SliderScalar("slider u64 high", ImGuiDataType_U64, &u64_v, &u64_hi_a, &u64_hi_b, "%I64u ms");
//		ImGui::SliderScalar("slider u64 full", ImGuiDataType_U64, &u64_v, &u64_min, &u64_max, "%I64u ms");
//		ImGui::SliderScalar("slider float low", ImGuiDataType_Float, &f32_v, &f32_zero, &f32_one);
//		ImGui::SliderScalar("slider float low^2", ImGuiDataType_Float, &f32_v, &f32_zero, &f32_one, "%.10f", 2.0f);
//		ImGui::SliderScalar("slider float high", ImGuiDataType_Float, &f32_v, &f32_lo_a, &f32_hi_a, "%e");
//		ImGui::SliderScalar("slider double low", ImGuiDataType_Double, &f64_v, &f64_zero, &f64_one, "%.10f grams", 1.0f);
//		ImGui::SliderScalar("slider double low^2", ImGuiDataType_Double, &f64_v, &f64_zero, &f64_one, "%.10f", 2.0f);
//		ImGui::SliderScalar("slider double high", ImGuiDataType_Double, &f64_v, &f64_lo_a, &f64_hi_a, "%e grams", 1.0f);
//
//		static bool inputs_step = true;
//		ImGui::Text("Inputs");
//		ImGui::Checkbox("Show step buttons", &inputs_step);
//		ImGui::InputScalar("input s8", ImGuiDataType_S8, &s8_v, inputs_step ? &s8_one : NULL, NULL, "%d");
//		ImGui::InputScalar("input u8", ImGuiDataType_U8, &u8_v, inputs_step ? &u8_one : NULL, NULL, "%u");
//		ImGui::InputScalar("input s16", ImGuiDataType_S16, &s16_v, inputs_step ? &s16_one : NULL, NULL, "%d");
//		ImGui::InputScalar("input u16", ImGuiDataType_U16, &u16_v, inputs_step ? &u16_one : NULL, NULL, "%u");
//		ImGui::InputScalar("input s32", ImGuiDataType_S32, &s32_v, inputs_step ? &s32_one : NULL, NULL, "%d");
//		ImGui::InputScalar("input s32 hex", ImGuiDataType_S32, &s32_v, inputs_step ? &s32_one : NULL, NULL, "%08X", ImGuiInputTextFlags_CharsHexadecimal);
//		ImGui::InputScalar("input u32", ImGuiDataType_U32, &u32_v, inputs_step ? &u32_one : NULL, NULL, "%u");
//		ImGui::InputScalar("input u32 hex", ImGuiDataType_U32, &u32_v, inputs_step ? &u32_one : NULL, NULL, "%08X", ImGuiInputTextFlags_CharsHexadecimal);
//		ImGui::InputScalar("input s64", ImGuiDataType_S64, &s64_v, inputs_step ? &s64_one : NULL);
//		ImGui::InputScalar("input u64", ImGuiDataType_U64, &u64_v, inputs_step ? &u64_one : NULL);
//		ImGui::InputScalar("input float", ImGuiDataType_Float, &f32_v, inputs_step ? &f32_one : NULL);
//		ImGui::InputScalar("input double", ImGuiDataType_Double, &f64_v, inputs_step ? &f64_one : NULL);
//
//		ImGui::TreePop();
//	}
//
//	if (ImGui::TreeNode("Multi-component Widgets"))
//	{
//		static float vec4f[4] = { 0.10f, 0.20f, 0.30f, 0.44f };
//		static int vec4i[4] = { 1, 5, 100, 255 };
//
//		ImGui::InputFloat2("input float2", vec4f);
//		ImGui::DragFloat2("drag float2", vec4f, 0.01f, 0.0f, 1.0f);
//		ImGui::SliderFloat2("slider float2", vec4f, 0.0f, 1.0f);
//		ImGui::InputInt2("input int2", vec4i);
//		ImGui::DragInt2("drag int2", vec4i, 1, 0, 255);
//		ImGui::SliderInt2("slider int2", vec4i, 0, 255);
//		ImGui::Spacing();
//
//		ImGui::InputFloat3("input float3", vec4f);
//		ImGui::DragFloat3("drag float3", vec4f, 0.01f, 0.0f, 1.0f);
//		ImGui::SliderFloat3("slider float3", vec4f, 0.0f, 1.0f);
//		ImGui::InputInt3("input int3", vec4i);
//		ImGui::DragInt3("drag int3", vec4i, 1, 0, 255);
//		ImGui::SliderInt3("slider int3", vec4i, 0, 255);
//		ImGui::Spacing();
//
//		ImGui::InputFloat4("input float4", vec4f);
//		ImGui::DragFloat4("drag float4", vec4f, 0.01f, 0.0f, 1.0f);
//		ImGui::SliderFloat4("slider float4", vec4f, 0.0f, 1.0f);
//		ImGui::InputInt4("input int4", vec4i);
//		ImGui::DragInt4("drag int4", vec4i, 1, 0, 255);
//		ImGui::SliderInt4("slider int4", vec4i, 0, 255);
//
//		ImGui::TreePop();
//	}
//
//	if (ImGui::TreeNode("Vertical Sliders"))
//	{
//		const float spacing = 4;
//		ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(spacing, spacing));
//
//		static int int_value = 0;
//		ImGui::VSliderInt("##int", ImVec2(18, 160), &int_value, 0, 5);
//		ImGui::SameLine();
//
//		static float values[7] = { 0.0f, 0.60f, 0.35f, 0.9f, 0.70f, 0.20f, 0.0f };
//		ImGui::PushID("set1");
//		for (int i = 0; i < 7; i++)
//		{
//			if (i > 0) ImGui::SameLine();
//			ImGui::PushID(i);
//			ImGui::PushStyleColor(ImGuiCol_FrameBg, (ImVec4)ImColor::HSV(i / 7.0f, 0.5f, 0.5f));
//			ImGui::PushStyleColor(ImGuiCol_FrameBgHovered, (ImVec4)ImColor::HSV(i / 7.0f, 0.6f, 0.5f));
//			ImGui::PushStyleColor(ImGuiCol_FrameBgActive, (ImVec4)ImColor::HSV(i / 7.0f, 0.7f, 0.5f));
//			ImGui::PushStyleColor(ImGuiCol_SliderGrab, (ImVec4)ImColor::HSV(i / 7.0f, 0.9f, 0.9f));
//			ImGui::VSliderFloat("##v", ImVec2(18, 160), &values[i], 0.0f, 1.0f, "");
//			if (ImGui::IsItemActive() || ImGui::IsItemHovered())
//				ImGui::SetTooltip("%.3f", values[i]);
//			ImGui::PopStyleColor(4);
//			ImGui::PopID();
//		}
//		ImGui::PopID();
//
//		ImGui::SameLine();
//		ImGui::PushID("set2");
//		static float values2[4] = { 0.20f, 0.80f, 0.40f, 0.25f };
//		const int rows = 3;
//		const ImVec2 small_slider_size(18, (160.0f - (rows - 1)*spacing) / rows);
//		for (int nx = 0; nx < 4; nx++)
//		{
//			if (nx > 0) ImGui::SameLine();
//			ImGui::BeginGroup();
//			for (int ny = 0; ny < rows; ny++)
//			{
//				ImGui::PushID(nx*rows + ny);
//				ImGui::VSliderFloat("##v", small_slider_size, &values2[nx], 0.0f, 1.0f, "");
//				if (ImGui::IsItemActive() || ImGui::IsItemHovered())
//					ImGui::SetTooltip("%.3f", values2[nx]);
//				ImGui::PopID();
//			}
//			ImGui::EndGroup();
//		}
//		ImGui::PopID();
//
//		ImGui::SameLine();
//		ImGui::PushID("set3");
//		for (int i = 0; i < 4; i++)
//		{
//			if (i > 0) ImGui::SameLine();
//			ImGui::PushID(i);
//			ImGui::PushStyleVar(ImGuiStyleVar_GrabMinSize, 40);
//			ImGui::VSliderFloat("##v", ImVec2(40, 160), &values[i], 0.0f, 1.0f, "%.2f\nsec");
//			ImGui::PopStyleVar();
//			ImGui::PopID();
//		}
//		ImGui::PopID();
//		ImGui::PopStyleVar();
//		ImGui::TreePop();
//	}
//
//	if (ImGui::TreeNode("Drag and Drop"))
//	{
//		{
//			// ColorEdit widgets automatically act as drag source and drag target.
//			// They are using standardized payload strings IMGUI_PAYLOAD_TYPE_COLOR_3F and IMGUI_PAYLOAD_TYPE_COLOR_4F to allow your own widgets
//			// to use colors in their drag and drop interaction. Also see the demo in Color Picker -> Palette demo.
//			ImGui::BulletText("Drag and drop in standard widgets");
//			ImGui::Indent();
//			static float col1[3] = { 1.0f, 0.0f, 0.2f };
//			static float col2[4] = { 0.4f, 0.7f, 0.0f, 0.5f };
//			ImGui::ColorEdit3("color 1", col1);
//			ImGui::ColorEdit4("color 2", col2);
//			ImGui::Unindent();
//		}
//
//		{
//			ImGui::BulletText("Drag and drop to copy/swap items");
//			ImGui::Indent();
//			enum Mode
//			{
//				Mode_Copy,
//				Mode_Move,
//				Mode_Swap
//			};
//			static int mode = 0;
//			if (ImGui::RadioButton("Copy", mode == Mode_Copy)) { mode = Mode_Copy; } ImGui::SameLine();
//			if (ImGui::RadioButton("Move", mode == Mode_Move)) { mode = Mode_Move; } ImGui::SameLine();
//			if (ImGui::RadioButton("Swap", mode == Mode_Swap)) { mode = Mode_Swap; }
//			static const char* names[9] = { "Bobby", "Beatrice", "Betty", "Brianna", "Barry", "Bernard", "Bibi", "Blaine", "Bryn" };
//			for (int n = 0; n < IM_ARRAYSIZE(names); n++)
//			{
//				ImGui::PushID(n);
//				if ((n % 3) != 0)
//					ImGui::SameLine();
//				ImGui::Button(names[n], ImVec2(60, 60));
//
//				// Our buttons are both drag sources and drag targets here!
//				if (ImGui::BeginDragDropSource(ImGuiDragDropFlags_None))
//				{
//					ImGui::SetDragDropPayload("DND_DEMO_CELL", &n, sizeof(int));        // Set payload to carry the index of our item (could be anything)
//					if (mode == Mode_Copy) { ImGui::Text("Copy %s", names[n]); }        // Display preview (could be anything, e.g. when dragging an image we could decide to display the filename and a small preview of the image, etc.)
//					if (mode == Mode_Move) { ImGui::Text("Move %s", names[n]); }
//					if (mode == Mode_Swap) { ImGui::Text("Swap %s", names[n]); }
//					ImGui::EndDragDropSource();
//				}
//				if (ImGui::BeginDragDropTarget())
//				{
//					if (const ImGuiPayload* payload = ImGui::AcceptDragDropPayload("DND_DEMO_CELL"))
//					{
//						IM_ASSERT(payload->DataSize == sizeof(int));
//						int payload_n = *(const int*)payload->Data;
//						if (mode == Mode_Copy)
//						{
//							names[n] = names[payload_n];
//						}
//						if (mode == Mode_Move)
//						{
//							names[n] = names[payload_n];
//							names[payload_n] = "";
//						}
//						if (mode == Mode_Swap)
//						{
//							const char* tmp = names[n];
//							names[n] = names[payload_n];
//							names[payload_n] = tmp;
//						}
//					}
//					ImGui::EndDragDropTarget();
//				}
//				ImGui::PopID();
//			}
//			ImGui::Unindent();
//		}
//
//		ImGui::TreePop();
//	}
//
//	if (ImGui::TreeNode("Querying Status (Active/Focused/Hovered etc.)"))
//	{
//		// Display the value of IsItemHovered() and other common item state functions. Note that the flags can be combined.
//		// (because BulletText is an item itself and that would affect the output of IsItemHovered() we pass all state in a single call to simplify the code).
//		static int item_type = 1;
//		static bool b = false;
//		static float col4f[4] = { 1.0f, 0.5, 0.0f, 1.0f };
//		static char str[16] = {};
//		ImGui::RadioButton("Text", &item_type, 0);
//		ImGui::RadioButton("Button", &item_type, 1);
//		ImGui::RadioButton("Checkbox", &item_type, 2);
//		ImGui::RadioButton("SliderFloat", &item_type, 3);
//		ImGui::RadioButton("InputText", &item_type, 4);
//		ImGui::RadioButton("InputFloat3", &item_type, 5);
//		ImGui::RadioButton("ColorEdit4", &item_type, 6);
//		ImGui::RadioButton("MenuItem", &item_type, 7);
//		ImGui::RadioButton("TreeNode (w/ double-click)", &item_type, 8);
//		ImGui::RadioButton("ListBox", &item_type, 9);
//		ImGui::Separator();
//		bool ret = false;
//		if (item_type == 0) { ImGui::Text("ITEM: Text"); }                                              // Testing text items with no identifier/interaction
//		if (item_type == 1) { ret = ImGui::Button("ITEM: Button"); }                                    // Testing button
//		if (item_type == 2) { ret = ImGui::Checkbox("ITEM: Checkbox", &b); }                            // Testing checkbox
//		if (item_type == 3) { ret = ImGui::SliderFloat("ITEM: SliderFloat", &col4f[0], 0.0f, 1.0f); }   // Testing basic item
//		if (item_type == 4) { ret = ImGui::InputText("ITEM: InputText", &str[0], IM_ARRAYSIZE(str)); }  // Testing input text (which handles tabbing)
//		if (item_type == 5) { ret = ImGui::InputFloat3("ITEM: InputFloat3", col4f); }                   // Testing multi-component items (IsItemXXX flags are reported merged)
//		if (item_type == 6) { ret = ImGui::ColorEdit4("ITEM: ColorEdit4", col4f); }                     // Testing multi-component items (IsItemXXX flags are reported merged)
//		if (item_type == 7) { ret = ImGui::MenuItem("ITEM: MenuItem"); }                                // Testing menu item (they use ImGuiButtonFlags_PressedOnRelease button policy)
//		if (item_type == 8) { ret = ImGui::TreeNodeEx("ITEM: TreeNode w/ ImGuiTreeNodeFlags_OpenOnDoubleClick", ImGuiTreeNodeFlags_OpenOnDoubleClick | ImGuiTreeNodeFlags_NoTreePushOnOpen); } // Testing tree node with ImGuiButtonFlags_PressedOnDoubleClick button policy.
//		if (item_type == 9) { const char* items[] = { "Apple", "Banana", "Cherry", "Kiwi" }; static int current = 1; ret = ImGui::ListBox("ITEM: ListBox", &current, items, IM_ARRAYSIZE(items), IM_ARRAYSIZE(items)); }
//		ImGui::BulletText(
//			"Return value = %d\n"
//			"IsItemFocused() = %d\n"
//			"IsItemHovered() = %d\n"
//			"IsItemHovered(_AllowWhenBlockedByPopup) = %d\n"
//			"IsItemHovered(_AllowWhenBlockedByActiveItem) = %d\n"
//			"IsItemHovered(_AllowWhenOverlapped) = %d\n"
//			"IsItemHovered(_RectOnly) = %d\n"
//			"IsItemActive() = %d\n"
//			"IsItemEdited() = %d\n"
//			"IsItemActivated() = %d\n"
//			"IsItemDeactivated() = %d\n"
//			"IsItemDeactivatedAfterEdit() = %d\n"
//			"IsItemVisible() = %d\n"
//			"IsItemClicked() = %d\n"
//			"GetItemRectMin() = (%.1f, %.1f)\n"
//			"GetItemRectMax() = (%.1f, %.1f)\n"
//			"GetItemRectSize() = (%.1f, %.1f)",
//			ret,
//			ImGui::IsItemFocused(),
//			ImGui::IsItemHovered(),
//			ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenBlockedByPopup),
//			ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenBlockedByActiveItem),
//			ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenOverlapped),
//			ImGui::IsItemHovered(ImGuiHoveredFlags_RectOnly),
//			ImGui::IsItemActive(),
//			ImGui::IsItemEdited(),
//			ImGui::IsItemActivated(),
//			ImGui::IsItemDeactivated(),
//			ImGui::IsItemDeactivatedAfterEdit(),
//			ImGui::IsItemVisible(),
//			ImGui::IsItemClicked(),
//			ImGui::GetItemRectMin().x, ImGui::GetItemRectMin().y,
//			ImGui::GetItemRectMax().x, ImGui::GetItemRectMax().y,
//			ImGui::GetItemRectSize().x, ImGui::GetItemRectSize().y
//			);
//
//		static bool embed_all_inside_a_child_window = false;
//		ImGui::Checkbox("Embed everything inside a child window (for additional testing)", &embed_all_inside_a_child_window);
//		if (embed_all_inside_a_child_window)
//			ImGui::BeginChild("outer_child", ImVec2(0, ImGui::GetFontSize() * 20), true);
//
//		// Testing IsWindowFocused() function with its various flags. Note that the flags can be combined.
//		ImGui::BulletText(
//			"IsWindowFocused() = %d\n"
//			"IsWindowFocused(_ChildWindows) = %d\n"
//			"IsWindowFocused(_ChildWindows|_RootWindow) = %d\n"
//			"IsWindowFocused(_RootWindow) = %d\n"
//			"IsWindowFocused(_AnyWindow) = %d\n",
//			ImGui::IsWindowFocused(),
//			ImGui::IsWindowFocused(ImGuiFocusedFlags_ChildWindows),
//			ImGui::IsWindowFocused(ImGuiFocusedFlags_ChildWindows | ImGuiFocusedFlags_RootWindow),
//			ImGui::IsWindowFocused(ImGuiFocusedFlags_RootWindow),
//			ImGui::IsWindowFocused(ImGuiFocusedFlags_AnyWindow));
//
//		// Testing IsWindowHovered() function with its various flags. Note that the flags can be combined.
//		ImGui::BulletText(
//			"IsWindowHovered() = %d\n"
//			"IsWindowHovered(_AllowWhenBlockedByPopup) = %d\n"
//			"IsWindowHovered(_AllowWhenBlockedByActiveItem) = %d\n"
//			"IsWindowHovered(_ChildWindows) = %d\n"
//			"IsWindowHovered(_ChildWindows|_RootWindow) = %d\n"
//			"IsWindowHovered(_ChildWindows|_AllowWhenBlockedByPopup) = %d\n"
//			"IsWindowHovered(_RootWindow) = %d\n"
//			"IsWindowHovered(_AnyWindow) = %d\n",
//			ImGui::IsWindowHovered(),
//			ImGui::IsWindowHovered(ImGuiHoveredFlags_AllowWhenBlockedByPopup),
//			ImGui::IsWindowHovered(ImGuiHoveredFlags_AllowWhenBlockedByActiveItem),
//			ImGui::IsWindowHovered(ImGuiHoveredFlags_ChildWindows),
//			ImGui::IsWindowHovered(ImGuiHoveredFlags_ChildWindows | ImGuiHoveredFlags_RootWindow),
//			ImGui::IsWindowHovered(ImGuiHoveredFlags_ChildWindows | ImGuiHoveredFlags_AllowWhenBlockedByPopup),
//			ImGui::IsWindowHovered(ImGuiHoveredFlags_RootWindow),
//			ImGui::IsWindowHovered(ImGuiHoveredFlags_AnyWindow));
//
//		ImGui::BeginChild("child", ImVec2(0, 50), true);
//		ImGui::Text("This is another child window for testing the _ChildWindows flag.");
//		ImGui::EndChild();
//		if (embed_all_inside_a_child_window)
//			ImGui::EndChild();
//
//		static char dummy_str[] = "This is a dummy field to be able to tab-out of the widgets above.";
//		ImGui::InputText("dummy", dummy_str, IM_ARRAYSIZE(dummy_str), ImGuiInputTextFlags_ReadOnly);
//
//		// Calling IsItemHovered() after begin returns the hovered status of the title bar.
//		// This is useful in particular if you want to create a context menu (with BeginPopupContextItem) associated to the title bar of a window.
//		static bool test_window = false;
//		ImGui::Checkbox("Hovered/Active tests after Begin() for title bar testing", &test_window);
//		if (test_window)
//		{
//			ImGui::Begin("Title bar Hovered/Active tests", &test_window);
//			if (ImGui::BeginPopupContextItem()) // <-- This is using IsItemHovered()
//			{
//				if (ImGui::MenuItem("Close")) { test_window = false; }
//				ImGui::EndPopup();
//			}
//			ImGui::Text(
//				"IsItemHovered() after begin = %d (== is title bar hovered)\n"
//				"IsItemActive() after begin = %d (== is window being clicked/moved)\n",
//				ImGui::IsItemHovered(), ImGui::IsItemActive());
//			ImGui::End();
//		}
//
//		ImGui::TreePop();
//	}
//}
