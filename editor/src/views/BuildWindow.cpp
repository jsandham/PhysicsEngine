#include "../../include/views/BuildWindow.h"

#include "core/Log.h"

#include "imgui.h"

#include <thread>

using namespace PhysicsEditor;

BuildWindow::BuildWindow() : PopupWindow("##BuildWindow", 600.0f, 300.0f, 500.0f, 200.0f)
{
	mTargetPlatform = TargetPlatform::Windows;
	mFilebrowser.setMode(FilebrowserMode::SelectFolder);
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

	static bool buildClicked = false;
	if (ImGui::Button("Build"))
	{
		buildClicked = true;
	}

	std::filesystem::path projectPath = clipboard.getProjectPath();
	std::filesystem::path buildPath = projectPath / "build";

	mFilebrowser.render(buildPath, buildClicked);

	if (mFilebrowser.isSelectFolderClicked())
	{
		build(mFilebrowser.getSelectedFolderPath());
		buildClicked = false;
		//std::thread t1(&BuildWindow::build, this, mFilebrowser.getSelectedFolderPath());
		//t1.join();
	}
}

void BuildWindow::build(const std::filesystem::path& path)
{
	std::filesystem::path buildPath = path / "build";
	std::filesystem::path buildGameDataPath = buildPath / "game_data";
	std::filesystem::path buildIncludePath = buildPath / "include";
	std::filesystem::path buildSrcPath = buildPath / "src";

	std::filesystem::create_directory(buildPath);
	std::filesystem::create_directory(buildGameDataPath);
	std::filesystem::create_directory(buildIncludePath);
	std::filesystem::create_directory(buildSrcPath);

	std::filesystem::copy_options copy_options = std::filesystem::copy_options::none;

	// Copy dll's to build path
	copy_options = std::filesystem::copy_options::overwrite_existing;
	std::filesystem::copy(std::filesystem::current_path() / "glew32.dll", buildPath, copy_options);
	std::filesystem::copy(std::filesystem::current_path() / "freetype.dll", buildPath, copy_options);

	// Copy data folder to build data path
	copy_options = std::filesystem::copy_options::recursive | std::filesystem::copy_options::overwrite_existing;
	std::filesystem::copy(path / "data", buildGameDataPath, copy_options);

	// Copy internal assets to build path
	std::filesystem::copy(std::filesystem::current_path() / "data", buildPath / "data", copy_options);

	// Copy include folder to build data path
	copy_options = std::filesystem::copy_options::recursive | std::filesystem::copy_options::overwrite_existing;

	// Copy src folder to build data path
	copy_options = std::filesystem::copy_options::recursive | std::filesystem::copy_options::overwrite_existing;
	std::filesystem::copy(std::filesystem::current_path() / "..\\..\\GameApplication.cpp", buildSrcPath, copy_options);
	std::filesystem::copy(std::filesystem::current_path() / "..\\..\\src\\Load.cpp", buildSrcPath, copy_options);

	std::filesystem::path executablePath = buildPath / "main.exe";
	std::filesystem::path buildScriptFilePath(std::filesystem::current_path() / "..\\..\\build.bat");

	std::string command = buildScriptFilePath.string() + " " +
		buildIncludePath.string() + " " +
		buildSrcPath.string() + " " +
		executablePath.string();

	//system(command.c_str());
}