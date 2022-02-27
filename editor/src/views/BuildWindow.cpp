#include "../../include/views/BuildWindow.h"

#include "core/Log.h"

#include "imgui.h"

#include <thread>
#include <chrono>
#include <iostream> 

using namespace PhysicsEditor;

#define BUILD_STEP(DESCRIPTION, STEP, INCREMENT) \
    mBuildStep = DESCRIPTION;       \
	STEP;                           \
	mBuildCompletion += INCREMENT
	
BuildWindow::BuildWindow() : PopupWindow("##BuildWindow", 600.0f, 300.0f, 500.0f, 200.0f)
{
	mTargetPlatform = TargetPlatform::Windows;
	mFilebrowser.setMode(FilebrowserMode::SelectFolder);

	mBuildCompletion = 0.0f;

	mWorker = std::thread(&BuildWindow::doWork, this);
}

BuildWindow::~BuildWindow()
{
	mWorker.join();
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

	if (clipboard.getWorld()->getNumberOfScenes() > 0)
	{
		size_t count = clipboard.getWorld()->getNumberOfScenes();
		PhysicsEngine::Scene* scene = clipboard.getWorld()->getSceneByIndex(0);
		PhysicsEngine::Guid currentSceneId = scene->getId();

		if (ImGui::BeginCombo("Starting Scene", scene->getName().c_str(), ImGuiComboFlags_None))
		{
			for (int i = 0; i < clipboard.getWorld()->getNumberOfScenes(); i++)
			{
				PhysicsEngine::Scene* s = clipboard.getWorld()->getSceneByIndex(i);

				std::string label = s->getName() + "##" + s->getId().toString();

				bool is_selected = (currentSceneId == s->getId());
				if (ImGui::Selectable(label.c_str(), is_selected))
				{
					currentSceneId = s->getId();
					scene = clipboard.getWorld()->getSceneById(currentSceneId);
				}
				if (is_selected)
				{
					ImGui::SetItemDefaultFocus();
				}
			}
			ImGui::EndCombo();
		}
	}

	static bool buildClicked = false;
	if (!buildClicked && ImGui::Button("Build"))
	{
		buildClicked = true;
	}
	
	std::filesystem::path projectPath = clipboard.getProjectPath();
	std::filesystem::path buildPath = projectPath / "build";

	mFilebrowser.render(buildPath, buildClicked);

	if (mFilebrowser.isSelectFolderClicked())
	{		
		buildClicked = false;
		mLaunchBuild = true;
		mBuildCompletion = 0.0f;
	}

	if (mBuildInProgress)
	{
		ImGui::Text(mBuildStep.c_str());
		ImGui::ProgressBar(mBuildCompletion);
	}

	//if (mBuildComplete)
	//{
	//	static bool okClicked = false;
	//	if (ImGui::Button("Ok"))
	//	{
	//		okClicked = true;
	//	}
	//}
}

void BuildWindow::build()
{
	mBuildInProgress = true;

	std::filesystem::path path = mFilebrowser.getSelectedFolderPath();

	std::filesystem::path buildPath = path / "build";
	std::filesystem::path buildGameDataPath = buildPath / "game_data";
	std::filesystem::path buildIncludePath = buildPath / "include";
	std::filesystem::path buildSrcPath = buildPath / "src";

	std::cout << "buildPath: " << buildPath << std::endl;
	std::cout << "buildGameDataPath: " << buildGameDataPath << std::endl;
	std::cout << "buildIncludePath: " << buildIncludePath << std::endl;
	std::cout << "buildSrcPath: " << buildSrcPath << std::endl;

	BUILD_STEP("Create build directories...", std::filesystem::create_directory(buildPath), 0.05f);
	BUILD_STEP("Create build directories...", std::filesystem::create_directory(buildGameDataPath), 0.05f);
	BUILD_STEP("Create build directories...", std::filesystem::create_directory(buildIncludePath), 0.05f);
	BUILD_STEP("Create build directories...", std::filesystem::create_directory(buildSrcPath), 0.05f);

	std::filesystem::copy_options copy_options = std::filesystem::copy_options::none;

	// Copy dll's to build path
	copy_options = std::filesystem::copy_options::overwrite_existing;
	BUILD_STEP("Copying dll files...", std::filesystem::copy(std::filesystem::current_path() / "glew32.dll", buildPath, copy_options), 0.05f);
	BUILD_STEP("Copying dll files...", std::filesystem::copy(std::filesystem::current_path() / "freetype.dll", buildPath, copy_options), 0.05f);

	// Copy data folder to build data path
	copy_options = std::filesystem::copy_options::recursive | std::filesystem::copy_options::overwrite_existing;
	BUILD_STEP("Copying data files...", std::filesystem::copy(path / "data", buildGameDataPath, copy_options), 0.05f);

	// Copy internal assets to build path
	BUILD_STEP("Copying data files...", std::filesystem::copy(std::filesystem::current_path() / "data", buildPath / "data", copy_options), 0.05f);

	// Copy include folder to build data path
	copy_options = std::filesystem::copy_options::recursive | std::filesystem::copy_options::overwrite_existing;

	// Copy src folder to build data path
	copy_options = std::filesystem::copy_options::recursive | std::filesystem::copy_options::overwrite_existing;
	BUILD_STEP("Copying source files...", std::filesystem::copy(std::filesystem::current_path() / "..\\..\\GameApplication.cpp", buildSrcPath, copy_options), 0.05f);
	BUILD_STEP("Copying source files...", std::filesystem::copy(std::filesystem::current_path() / "..\\..\\src\\Load.cpp", buildSrcPath, copy_options), 0.05f);

	std::filesystem::path executablePath = buildPath / "main.exe";
	std::filesystem::path buildScriptFilePath(std::filesystem::current_path() / "..\\..\\build.bat");

	std::string command = buildScriptFilePath.string() + " " +
		buildIncludePath.string() + " " +
		buildSrcPath.string() + " " +
		executablePath.string();

	std::cout << "command: " << command << std::endl;

	BUILD_STEP("Compile game...", system(command.c_str()), 0.5f);

	mBuildInProgress = false;
}


void BuildWindow::doWork()
{
	while (true)
	{
		if (mLaunchBuild)
		{
			mLaunchBuild = false;
			build();
		}

		std::this_thread::sleep_for(std::chrono::milliseconds(200));
	}
}
