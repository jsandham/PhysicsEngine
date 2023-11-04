#include "../../include/views/AboutPopup.h"

#include <core/SystemQuery.h>

#include "imgui.h"

using namespace PhysicsEditor;

AboutPopup::AboutPopup()
{
}

AboutPopup::~AboutPopup()
{
}

void AboutPopup::init(Clipboard& clipboard)
{
}

void AboutPopup::update(Clipboard& clipboard, bool isOpenedThisFrame)
{
	if (isOpenedThisFrame)
	{
		ImGui::SetNextWindowPos(ImVec2(600.0f, 300.0f));
		ImGui::SetNextWindowSize(ImVec2(300.0f, 300.0f));

		ImGui::OpenPopup("About");
		mOpen = true;
	}

	if (ImGui::BeginPopupModal("About", &mOpen, ImGuiWindowFlags_NoResize))
	{
		ImGui::Text("About PhysicsEngine");
		ImGui::TextWrapped("About engine text goes here");

		ImGui::Text(ImGui::GetVersion());
		ImGui::Text(IMGUI_VERSION);
		ImGui::Text(std::to_string(IMGUI_VERSION_NUM).c_str());

		PhysicsEngine::CPUInfo info;
		PhysicsEngine::queryCpuInfo(&info);

		std::string archStr = "Processor Architecture: " + std::string(PhysicsEngine::ProcessorArchToString(info.arch));
		std::string numCpuCoresStr = "Number of CPU cores: " + std::to_string(info.numCpuCores);
		std::string pageSizeStr = "Page Size: " + std::to_string(info.pageSize);
		std::string openmpEnabledStr = "OpenMP Enabled? " + std::to_string(info.openmpEnabled);
		std::string openmpMaxThreadsStr = "OpenMP maximum threads: " + std::to_string(info.openmp_max_threads);

		ImGui::Text(archStr.c_str());
		ImGui::Text(numCpuCoresStr.c_str());
		ImGui::Text(pageSizeStr.c_str());
		ImGui::Text(openmpEnabledStr.c_str());
		ImGui::Text(openmpMaxThreadsStr.c_str());

		if (ImGui::Button("Ok"))
		{
			ImGui::CloseCurrentPopup();
		}

		ImGui::EndPopup();
	}
}