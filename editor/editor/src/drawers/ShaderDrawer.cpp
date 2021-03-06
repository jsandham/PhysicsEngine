#include <iostream>
#include <string>
#include <vector>

#include "../../include/Undo.h"
#include "../../include/EditorCommands.h"
#include "../../include/drawers/ShaderDrawer.h"

#include "core/Shader.h"

#include "imgui.h"
#include "imgui_impl_opengl3.h"
#include "imgui_impl_win32.h"
#include "imgui_internal.h"

using namespace PhysicsEditor;

ShaderDrawer::ShaderDrawer()
{
}

ShaderDrawer::~ShaderDrawer()
{
}

void ShaderDrawer::render(EditorClipboard &clipboard, Guid id)
{
    Shader *shader = clipboard.getWorld()->getAssetById<Shader>(id);

    std::vector<ShaderProgram> programs = shader->getPrograms();
    std::vector<ShaderUniform> uniforms = shader->getUniforms();

    // display basic shader information
    ImGui::Text("Info:");
    ImGui::Dummy(ImVec2(0.0f, 5.0f));

    ImGui::Columns(2, "InfoColumns", false);
    ImGui::Text("Version");
    ImGui::Text("Variants");

    ImGui::NextColumn();

    ImGui::Text("OpenGL 3.3");
    ImGui::Text(std::to_string(programs.size()).c_str());

    ImGui::Columns(1);

    ImGui::Dummy(ImVec2(0.0f, 20.0f));
    ImGui::Separator();

    // display shader uniforms
    ImGui::Text("Uniforms:");
    ImGui::Dummy(ImVec2(0.0f, 5.0f));

    ImGui::Columns(2, "UniformColumns", false);

    for (size_t i = 0; i < uniforms.size(); i++)
    {
        if (std::strcmp(uniforms[i].mBlockName, "material") == 0)
        {
            ImGui::Text(uniforms[i].mShortName);
        }
    }

    ImGui::NextColumn();

    for (size_t i = 0; i < uniforms.size(); i++)
    {
        if (std::strcmp(uniforms[i].mBlockName, "material") == 0)
        {
            ImGui::Text(std::to_string(uniforms[i].mType).c_str());
        }
    }

    ImGui::Dummy(ImVec2(0.0f, 20.0f));
    ImGui::Columns(1);
}