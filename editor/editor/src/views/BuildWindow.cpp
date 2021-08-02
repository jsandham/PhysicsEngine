#include "../../include/views/BuildWindow.h"

#include "core/Log.h"

#include "imgui.h"

using namespace PhysicsEditor;

BuildWindow::BuildWindow() : PopupWindow("##BuildWindow", 600.0f, 300.0f, 500.0f, 200.0f)
{
	mTargetPlatform = TargetPlatform::Windows;
}

BuildWindow::~BuildWindow()
{
}

void BuildWindow::init(Clipboard &clipboard)
{
}

void BuildWindow::update(Clipboard &clipboard)
{
	int targetPlatformIndex = static_cast<int>(mTargetPlatform);
	const char* targetPlatformNames[] = { "Windows", "Linux" };

	if (ImGui::Combo("Target Platform", &targetPlatformIndex, targetPlatformNames, 2))
	{
		mTargetPlatform = static_cast<TargetPlatform>(targetPlatformIndex);
	}

	if (ImGui::Button("Build"))
	{
		std::filesystem::path path = clipboard.getProjectPath() / "build";

		if (std::filesystem::create_directory(path))
		{
			bool success = true;
			success &= std::filesystem::create_directory(path / "data");

		}

		std::filesystem::path compilerPath("C:\\Program Files\\LLVM\\bin\\clang-cl");
		std::filesystem::path sourcePath("C:\\Users\\jsand\\Documents\\main.cpp");
		std::filesystem::path executablePath("C:\\Users\\jsand\\Documents\\main.exe");

		std::filesystem::path buildPath("C:\\Users\\jsand\\Documents\\PhysicsEngine\\editor\\build.bat");

		std::string command = compilerPath.string() + " " + sourcePath.string() + " -o " + executablePath.string();

		int test = system(buildPath.string().c_str());
		//int test = system("dir");

		PhysicsEngine::Log::info(("test " + std::to_string(test) + "\n").c_str());
		PhysicsEngine::Log::info(("buildPath " + buildPath.string() + "\n").c_str());
		PhysicsEngine::Log::info(("compilerPath " + compilerPath.string() + "\n").c_str());
		PhysicsEngine::Log::info(("sourcePath " + sourcePath.string() + "\n").c_str());
		PhysicsEngine::Log::info(("executablePath " + executablePath.string() + "\n").c_str());
		PhysicsEngine::Log::info(("command " + command + "\n").c_str());
		//PhysicsEngine::Log::info(("test " + std::filesystem::current_path().string() + "\n").c_str());
	}
}