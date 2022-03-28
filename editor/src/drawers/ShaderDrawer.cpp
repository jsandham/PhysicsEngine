#include <iostream>
#include <string>
#include <vector>

#include "../../include/drawers/ShaderDrawer.h"
#include "../../include/imgui/imgui_extensions.h"

#include "core/Shader.h"
#include "core/Log.h"

#include "imgui.h"
#include "glm/glm.hpp"

using namespace PhysicsEditor;

ShaderDrawer::ShaderDrawer()
{
}

ShaderDrawer::~ShaderDrawer()
{
}

void ShaderDrawer::render(Clipboard &clipboard, const Guid& id)
{
    InspectorDrawer::render(clipboard, id);

    ImGui::Separator();
    mContentMin = ImGui::GetItemRectMin();

    Shader *shader = clipboard.getWorld()->getAssetById<Shader>(id);

    std::vector<ShaderProgram> programs = shader->getPrograms();
    std::vector<ShaderUniform> uniforms = shader->getUniforms();

    if(ImGui::BeginTable("Shader Info", 2))
    {
        ImGui::TableNextColumn();
        ImGui::Text("Shader Info");
        ImGui::TableNextColumn();

        ImGui::TableNextColumn();
        ImGui::Text("Name:");
        ImGui::TableNextColumn();
        ImGui::Text(shader->getName().c_str());

        ImGui::TableNextColumn();
        ImGui::Text("Language:");
        ImGui::TableNextColumn();
        switch (shader->getSourceLanguage())
        {
        case ShaderSourceLanguage::GLSL: {ImGui::Text("GLSL"); break; }
        case ShaderSourceLanguage::HLSL: {ImGui::Text("HLSL"); break; }
        }

        ImGui::TableNextColumn();
        ImGui::Text("Source:");
        ImGui::TableNextColumn();
        ImGui::Text(shader->getSource().c_str());

        ImGui::TableNextColumn();
        ImGui::Text("Variant Count:");
        ImGui::TableNextColumn();
        ImGui::Text(std::to_string(programs.size()).c_str());

        ImGui::TableNextColumn();
        ImGui::Text("Uniform Count:");
        ImGui::TableNextColumn();
        ImGui::Text(std::to_string(uniforms.size()).c_str());

        ImGui::EndTable();
    }

    ImGui::Separator();

    /*bool static showVariants = false;
    if (ImGui::Checkbox("Show Variants", &showVariants))
    {
    }
    if(showVariants)
    {
        size_t currentProgram = 0;
        char currentText[16384];

        if (ImGui::BeginCombo("Variant", "0", ImGuiComboFlags_None))
        {
            for (size_t i = 0; i < programs.size(); i++)
            {
                std::string label = std::to_string(i) + "##" + shader->getId().toString();

                bool is_selected = (currentProgram == i);
                if (ImGui::Selectable(label.c_str(), is_selected))
                {
                    currentProgram = i;
                    strncpy(currentText, programs[i].mVertexShader.c_str(), sizeof(currentText));
                }
                if (is_selected)
                {
                    ImGui::SetItemDefaultFocus();
                }
            }
            ImGui::EndCombo();
        }

        ImGui::InputTextMultiline("Shader Variant", currentText, 16384, ImVec2(-FLT_MIN, ImGui::GetTextLineHeight() * 32), ImGuiInputTextFlags_ReadOnly);
    }*/

    static bool vertexShaderActive = true;
    static bool geometryShaderActive = false;
    static bool fragmentShaderActive = false;

    // select shader stage to view
    ImGui::SetCursorPosX(glm::max(0.5f * ImGui::GetWindowSize().x - 100.0f, 0.0f));
    ImGui::BeginGroup();
    if (ImGui::StampButton("Vertex", vertexShaderActive))
    {
        vertexShaderActive = true;
        geometryShaderActive = false;
        fragmentShaderActive = false;
    }
    ImGui::SameLine();

    if (ImGui::StampButton("Geometry", geometryShaderActive))
    {
        vertexShaderActive = false;
        geometryShaderActive = true;
        fragmentShaderActive = false;
    }
    ImGui::SameLine();

    if (ImGui::StampButton("Fragment", fragmentShaderActive))
    {
        vertexShaderActive = false;
        geometryShaderActive = false;
        fragmentShaderActive = true;
    }
    ImGui::SameLine();
    ImGui::EndGroup();

    static char* text = nullptr;

    static char vertexText[16384];
    static char fragmentText[16384];
    static char geometryText[16384];

    static bool vertexTextFilled = false;
    static bool fragmentTextFilled = false;
    static bool geometryTextFilled = false;

    // Fill buffers with current shaders
    if (!vertexTextFilled)
    {
        strncpy(vertexText, shader->getVertexShader().c_str(), sizeof(vertexText));
        vertexText[sizeof(vertexText) - 1] = 0;
        vertexTextFilled = true;
    }
    if (!fragmentTextFilled)
    {
        strncpy(fragmentText, shader->getFragmentShader().c_str(), sizeof(fragmentText));
        fragmentText[sizeof(fragmentText) - 1] = 0;
        fragmentTextFilled = true;
    }
    if (!geometryTextFilled)
    {
        strncpy(geometryText, shader->getGeometryShader().c_str(), sizeof(geometryText));
        geometryText[sizeof(geometryText) - 1] = 0;
        geometryTextFilled = true;
    }

    if (vertexShaderActive)
    {
        text = &vertexText[0];
    }
    else if (fragmentShaderActive)
    {
        text = &fragmentText[0];
    }
    else
    {
        text = &geometryText[0];
    }

    assert(text != nullptr);

    static ImGuiInputTextFlags flags = ImGuiInputTextFlags_AllowTabInput;
    ImGui::InputTextMultiline("Shader Source", text, 16384, ImVec2(-FLT_MIN, ImGui::GetTextLineHeight() * 32), flags);

    if (ImGui::BeginTable("Compilation Status", 2))
    {
        // Determine if shader has been edited
        bool vertexEditMade = (strcmp(shader->getVertexShader().c_str(), &vertexText[0]) != 0);
        bool fragmentEditMade = (strcmp(shader->getFragmentShader().c_str(), &fragmentText[0]) != 0);
        bool geometryEditMade = (strcmp(shader->getGeometryShader().c_str(), &geometryText[0]) != 0);

        // Determine if all shader variants compiled successfully
        bool vertexCompiled = true;
        bool fragmentCompiled = true;
        bool geometryCompiled = true;
        bool linked = true;
        for (size_t i = 0; i < programs.size(); i++)
        {
            vertexCompiled &= programs[i].mStatus.mVertexShaderCompiled;
            fragmentCompiled &= programs[i].mStatus.mFragmentShaderCompiled;
            geometryCompiled &= programs[i].mStatus.mGeometryShaderCompiled;
            linked &= programs[i].mStatus.mShaderLinked;
        }

        ImGui::TableNextColumn();
        ImGui::Text("Vertex Shader:");
        ImGui::TableNextColumn();
        if (vertexEditMade)
        {
            ImGui::TextColored(ImVec4(1.0f, 0.91764705f, 0.01568627f, 1.0f), "Recompile required");
        }
        else
        {
            if (vertexCompiled)
            {
                ImGui::TextColored(ImVec4(0.0f, 1.0f, 0.0f, 1.0f), "Success");
            }
            else
            {
                ImGui::TextColored(ImVec4(1.0f, 0.0f, 0.0f, 1.0f), "Failed");
            }
        }

        ImGui::TableNextColumn();
        ImGui::Text("Fragment Shader:");
        ImGui::TableNextColumn();
        if (fragmentEditMade)
        {
            ImGui::TextColored(ImVec4(1.0f, 0.91764705f, 0.01568627f, 1.0f), "Recompile required");
        }
        else
        {
            if (fragmentCompiled)
            {
                ImGui::TextColored(ImVec4(0.0f, 1.0f, 0.0f, 1.0f), "Success");
            }
            else
            {
                ImGui::TextColored(ImVec4(1.0f, 0.0f, 0.0f, 1.0f), "Failed");
            }
        }

        ImGui::TableNextColumn();
        ImGui::Text("Geometry Shader:");
        ImGui::TableNextColumn();
        if (geometryEditMade)
        {
            ImGui::TextColored(ImVec4(1.0f, 0.91764705f, 0.01568627f, 1.0f), "Recompile required");
        }
        else
        {
            if (shader->getGeometryShader().empty())
            {
                ImGui::TextColored(ImVec4(1.0f, 1.0f, 1.0f, 1.0f), "-");
            }
            else if (geometryCompiled)
            {
                ImGui::TextColored(ImVec4(0.0f, 1.0f, 0.0f, 1.0f), "Success");
            }
            else
            {
                ImGui::TextColored(ImVec4(1.0f, 0.0f, 0.0f, 1.0f), "Failed");
            }
        }

        ImGui::TableNextColumn();
        ImGui::Text("Linking:");
        ImGui::TableNextColumn();
        if (linked)
        {
            ImGui::TextColored(ImVec4(0.0f, 1.0f, 0.0f, 1.0f), "Success");
        }
        else
        {
            ImGui::TextColored(ImVec4(1.0f, 0.0f, 0.0f, 1.0f), "Failed");
        }

        ImGui::EndTable();
    }

    //ImGui::Separator();

    ImGui::SetCursorPosX(glm::max(ImGui::GetWindowSize().x - 80.0f, 0.0f));
    if (ImGui::Button("Recompile"))
    {
        if (vertexShaderActive)
        {
            shader->setVertexShader(std::string(text));
        }
        else if (fragmentShaderActive)
        {
            shader->setFragmentShader(std::string(text));
        }
        else
        {
            shader->setGeometryShader(std::string(text));
        }
    }

    ImGui::Separator();


    //if (ImGui::Button("Reload from file"))
    //{
    //    vertexTextFilled = false;
    //    fragmentTextFilled = false;
    //    geometryTextFilled = false;
    //}

    //if (ImGui::Button("Save to file"))
    //{
    //
    //}

    mContentMax = ImGui::GetItemRectMax();
}