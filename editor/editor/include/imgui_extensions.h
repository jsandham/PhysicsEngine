#ifndef __IMGUI_EXTENSIONS_H__
#define __IMGUI_EXTENSIONS_H__

#include <vector>
#include <string>

namespace ImGui
{

	bool BeginDropdown(std::string name, std::vector<std::string> values, int* selection);
	void EndDropdown();

	void EnableDocking();
}

#endif
