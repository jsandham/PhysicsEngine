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
		std::filesystem::path buildPath = clipboard.getProjectPath() / "build";

		if (std::filesystem::create_directory(buildPath))
		{
			bool success = std::filesystem::create_directory(buildPath / "data");

			if (!success)
			{
				PhysicsEngine::Log::error("Could not create build directory\n");
				return;
			}
		}

		std::filesystem::copy_options copy_options = std::filesystem::copy_options::overwrite_existing;
		std::filesystem::copy(std::filesystem::current_path() / "..\\x64\\Debug\\glew32.dll", buildPath, copy_options);
		std::filesystem::copy(std::filesystem::current_path() / "..\\x64\\Debug\\freetype.dll", buildPath, copy_options);

		std::filesystem::path sourcePath = std::filesystem::current_path() / "..\\..\\engine\\src\\core\\platform\\main_win32.cpp";
		std::filesystem::path executablePath = buildPath / "main.exe";
		std::filesystem::path compilerPath("C:\\Program Files\\LLVM\\bin\\clang-cl");
		std::filesystem::path buildScriptFilePath(std::filesystem::current_path() / "..\\build.bat");

		std::string command = buildScriptFilePath.string() + " " + sourcePath.string() + " " + executablePath.string();

		int test = system(command.c_str());







		PhysicsEngine::Log::info(("test " + std::to_string(test) + "\n").c_str());
		PhysicsEngine::Log::info(("buildPath " + buildPath.string() + "\n").c_str());
		PhysicsEngine::Log::info(("compilerPath " + compilerPath.string() + "\n").c_str());
		PhysicsEngine::Log::info(("sourcePath " + sourcePath.string() + "\n").c_str());
		PhysicsEngine::Log::info(("executablePath " + executablePath.string() + "\n").c_str());
		PhysicsEngine::Log::info(("command " + command + "\n").c_str());
		PhysicsEngine::Log::info(("cwd " + std::filesystem::current_path().string() + "\n").c_str());
	}
}