#include "../../include/views/AboutPopup.h"

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


		if (ImGui::Button("Ok"))
		{
			ImGui::CloseCurrentPopup();
		}

		ImGui::EndPopup();
	}
}