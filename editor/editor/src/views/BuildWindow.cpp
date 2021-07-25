#include "../../include/views/BuildWindow.h"

#include "core/Log.h"

#include "imgui.h"

using namespace PhysicsEditor;

BuildWindow::BuildWindow()
{
}

BuildWindow::~BuildWindow()
{
}

void BuildWindow::init(Clipboard &clipboard)
{
}

void BuildWindow::update(Clipboard &clipboard)
{
	if (ImGui::Button("Build"))
	{
		int test = system("\"C:\\Program Files\\LLVM\\bin\\clang-cl\" \"C:\\Users\\jsand\\Documents\\main.cpp\" -o \"C:\\Users\\jsand\\Documents\\main.exe\"");
		//int test = system("dir");
	
		PhysicsEngine::Log::info(("test " + std::to_string(test) + "\n").c_str());
	}
}