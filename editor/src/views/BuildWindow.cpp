#include "../../include/views/BuildWindow.h"

#include "core/Log.h"

#include <thread>
#include <chrono>
#include <iostream>
#include <stdexcept>
#include <stdio.h>
#include <string>
#include <filesystem>

using namespace PhysicsEditor;

#define BUILD_STEP(DESCRIPTION, STEP, FRACTION) \
    mBuildStep = DESCRIPTION;                   \
	STEP;                                       \
	mBuildCompletion = FRACTION

std::string exec(const char* cmd) {
	char buffer[16384];
	std::string result = "";
	FILE* pipe = _popen(cmd, "r");
	if (!pipe) throw std::runtime_error("popen() failed!");
	try {
		while (fgets(buffer, sizeof buffer, pipe) != NULL) {
			result += buffer;
		}
	}
	catch (...) {
		_pclose(pipe);
		throw;
	}
	_pclose(pipe);
	return result;
}

BuildWindow::BuildWindow()
{
	mName = "##BuildWindow";
	mX = 600.0f;
	mY = 300.0f;
	mWidth = 400.0f;
	mHeight = 200.0f;
	mOpen = true;

	mTargetPlatform = TargetPlatform::Windows;
	mFilebrowser.setMode(FilebrowserMode::SelectFolder);

	mBuildCompletion = 0.0f;

	mBuildWorker = std::thread(&BuildWindow::doWork, this);
}

BuildWindow::BuildWindow(const std::string& name, float x, float y, float width, float height)
{
	mName = name;
	mX = x;
	mY = y;
	mWidth = width;
	mHeight = height;
	mOpen = true;

	mTargetPlatform = TargetPlatform::Windows;
	mFilebrowser.setMode(FilebrowserMode::SelectFolder);

	mBuildCompletion = 0.0f;

	mBuildWorker = std::thread(&BuildWindow::doWork, this);
}

BuildWindow::~BuildWindow()
{
	mBuildWorker.join();
}

void BuildWindow::draw(Clipboard& clipboard, bool isOpenedThisFrame)
{
	if (isOpenedThisFrame)
	{
		ImGui::SetNextWindowPos(ImVec2(mX, mY));
		ImGui::SetNextWindowSize(ImVec2(mWidth, mHeight));

		ImGui::OpenPopup(mName.c_str());
		mOpen = true;
	}

	if (ImGui::BeginPopupModal(mName.c_str(), &mOpen, ImGuiWindowFlags_NoResize))
	{
		update(clipboard);

		ImGui::EndPopup();
	}

	if (!mOpen)
	{
		if (mBuildComplete)
		{
			mBuildInProgress = false;
			mBuildComplete = false;
		}
	}
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

	/*if (clipboard.getWorld()->getNumberOfScenes() > 0)
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
	}*/

	static bool buildClicked = false;
	if (!mBuildInProgress && ImGui::Button("Build"))
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

		mBuildLog.Draw("Build Log", ImVec2(mWidth, mHeight / 2.5f), true);
	}

	if (mBuildComplete)
	{
		if (ImGui::Button("Ok"))
		{
			mBuildInProgress = false;
			mBuildComplete = false;

			mOpen = false;
		}
	}
}

void BuildWindow::build()
{
	mBuildInProgress = true;

	mBuildLog.Clear();

	std::filesystem::path path = mFilebrowser.getSelectedFolderPath();

	std::filesystem::path buildPath = path / "build";
	std::filesystem::path buildGameDataPath = buildPath / "game_data";
	std::filesystem::path buildIncludePath = buildPath / "include";
	std::filesystem::path buildSrcPath = buildPath / "src";

	BUILD_STEP("Create build directories...", std::filesystem::create_directory(buildPath), 0.05f);
	BUILD_STEP("Create build directories...", std::filesystem::create_directory(buildGameDataPath), 0.1f);
	BUILD_STEP("Create build directories...", std::filesystem::create_directory(buildIncludePath), 0.15f);
	BUILD_STEP("Create build directories...", std::filesystem::create_directory(buildSrcPath), 0.2f);

	mBuildLog.AddLog("Create build directories...\n");

	std::filesystem::copy_options copy_options = std::filesystem::copy_options::none;

	// Copy dll's to build path
	copy_options = std::filesystem::copy_options::overwrite_existing;
	BUILD_STEP("Copying dll files...", std::filesystem::copy(std::filesystem::current_path() / "glew32.dll", buildPath, copy_options), 0.25f);
	BUILD_STEP("Copying dll files...", std::filesystem::copy(std::filesystem::current_path() / "freetype.dll", buildPath, copy_options), 0.3f);

	mBuildLog.AddLog("Copying dll files...\n");

	// Copy data folder to build data path
	copy_options = std::filesystem::copy_options::recursive | std::filesystem::copy_options::overwrite_existing;
	BUILD_STEP("Copying data files...", std::filesystem::copy(path / "data", buildGameDataPath, copy_options), 0.35f);

	// Copy internal assets to build path
	BUILD_STEP("Copying data files...", std::filesystem::copy(std::filesystem::current_path() / "data", buildPath / "data", copy_options), 0.4f);

	// Copy include folder to build data path
	copy_options = std::filesystem::copy_options::recursive | std::filesystem::copy_options::overwrite_existing;

	// Copy src folder to build data path
	copy_options = std::filesystem::copy_options::recursive | std::filesystem::copy_options::overwrite_existing;
	BUILD_STEP("Copying source files...", std::filesystem::copy(std::filesystem::current_path() / "..\\..\\GameApplication.cpp", buildSrcPath, copy_options), 0.45f);
	BUILD_STEP("Copying source files...", std::filesystem::copy(std::filesystem::current_path() / "..\\..\\src\\Load.cpp", buildSrcPath, copy_options), 0.5f);

	mBuildLog.AddLog("Copying source files...\n");

	std::filesystem::path executablePath = buildPath / "main.exe";
	
	const std::string ENGINE_INC = "..\\..\\..\\engine\\include";
	const std::string YAML_INC = "..\\..\\..\\external\\yaml-cpp\\include";
	const std::string GLEW_INC = "..\\..\\..\\external\\glew-2.1.0";
	const std::string FREETYPE_INC = "..\\..\\..\\external\\freetype";
	const std::string GLM_INC = "..\\..\\..\\external\\glm";

	const std::string INCLUDES = "/I" + ENGINE_INC + " /I" + YAML_INC + " /I" + GLEW_INC + " /I" + FREETYPE_INC + " /I" + GLM_INC;

	const std::string ENGINE_LIB = "..\\..\\..\\engine\\lib\\debug\\engine.lib";
	const std::string YAML_LIB = "..\\..\\..\\external\\yaml-cpp\\lib\\debug\\yaml-cppd.lib";
	const std::string GLEW_LIB = "..\\..\\..\\engine\\lib\\debug\\glew32.lib";
	const std::string FREETYPE_LIB = "..\\..\\..\\engine\\lib\\debug\\freetype.lib";

	const std::string LIBS = "kernel32.lib user32.lib gdi32.lib ole32.lib opengl32.lib " + ENGINE_LIB + " " + YAML_LIB + " " + GLEW_LIB + " " + FREETYPE_LIB;
	
	const std::string OPT = "/Od";
	const std::string WARN = "-W4 -wd4100 -wd4996 -wd4211";
	const std::string FLAGS = "/MDd -Zi -nologo /EHsc";

	const std::string INCLUDE_PATH = buildIncludePath.string();
	const std::string SOURCE_PATH = buildSrcPath.string();
	const std::string EXECUTABLE = executablePath.string();
	const std::string COMPILER = "cl";

	std::vector<std::string> includeFiles;
	std::string INCLUDE_FILES = "";
	std::error_code error_code;
	for (const std::filesystem::directory_entry& entry : std::filesystem::recursive_directory_iterator(INCLUDE_PATH, error_code))
	{
		if (std::filesystem::is_regular_file(entry, error_code))
		{
			includeFiles.push_back(entry.path().string());
			if(!INCLUDE_FILES.empty())
			{
				INCLUDE_FILES += " ";
			}
			INCLUDE_FILES += "/I" + entry.path().string();
		}
	}

	std::vector<std::string> srcFiles;
	std::string SRC_FILES = "";
	for (const std::filesystem::directory_entry& entry : std::filesystem::recursive_directory_iterator(SOURCE_PATH, error_code))
	{
		if (std::filesystem::is_regular_file(entry, error_code))
		{
			srcFiles.push_back(entry.path().string());
			if (!SRC_FILES.empty())
			{
				SRC_FILES += " ";
			}
			SRC_FILES += entry.path().string();
		}
	}

	/*for (size_t i = 0; i < srcFiles.size(); i++)
	{
		std::string command = "..\\..\\..\\shell.bat && " + COMPILER + " /std:c++17 -c " + srcFiles[i] + " " + INCLUDE_FILES + " " + INCLUDES + " " + OPT + " " + WARN + " " + FLAGS;
		BUILD_STEP("Compile file...", exec(command.c_str()), 0.0f);
	}*/

	const std::string command = "..\\..\\..\\shell.bat && " + 
								COMPILER + " /std:c++17 -o " + 
								EXECUTABLE + " " + 
								SRC_FILES + " " +
								INCLUDE_FILES + " " + 
								INCLUDES + " " + 
								OPT + " " + 
								WARN + " " + 
								FLAGS + " " + 
								LIBS;

	BUILD_STEP("Compile game...", mBuildLog.AddLog(exec(command.c_str()).c_str()), 0.9f);
	
	BUILD_STEP("Done!", mBuildLog.AddLog("Done!\n"), 1.0f);

	mBuildComplete = true;
}

void BuildWindow::doWork()
{
	while (true)
	{
		if (mLaunchBuild)
		{
			mLaunchBuild = false;
			
			try
			{
				build();
			}
			catch (...)
			{
				mBuildComplete = true;
			}
		}

		std::this_thread::sleep_for(std::chrono::milliseconds(200));
	}
}
