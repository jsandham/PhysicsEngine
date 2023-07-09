#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include "../../include/drawers/ShaderDrawer.h"
#include "../../include/imgui/imgui_extensions.h"

#include "core/Shader.h"
#include "core/Log.h"
#include "graphics/RenderContext.h"

#include "imgui.h"
#include "glm/glm.hpp"

using namespace PhysicsEditor;

ShaderDrawer::ShaderDrawer()
{
    mShaderId = PhysicsEngine::Guid::INVALID;
}

ShaderDrawer::~ShaderDrawer()
{
}

void ShaderDrawer::render(Clipboard &clipboard, const PhysicsEngine::Guid& id)
{
    ImGui::Separator();
    mContentMin = ImGui::GetItemRectMin();

    PhysicsEngine::Shader *shader = clipboard.getWorld()->getAssetByGuid<PhysicsEngine::Shader>(id);

    if (shader != nullptr)
    {
        std::vector<PhysicsEngine::ShaderProgram*> programs = shader->getPrograms();

        if (ImGui::BeginTable("Shader Info", 2))
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
            ImGui::Text(PhysicsEngine::GetShaderLanguageStringFromRenderAPI(PhysicsEngine::RenderContext::getRenderAPI()));

            ImGui::TableNextColumn();
            ImGui::Text("Source:");
            ImGui::TableNextColumn();
            ImGui::Text(shader->getSource().c_str());
            
            ImGui::EndTable();
        }

        ImGui::Separator();

        {
            std::vector<PhysicsEngine::ShaderUniform> uniforms = shader->getUniforms();
            static ImGuiTableFlags flags = ImGuiTableFlags_ScrollY | 
                                           ImGuiTableFlags_Resizable | 
                                           ImGuiTableFlags_Reorderable | 
                                           ImGuiTableFlags_Hideable | 
                                           ImGuiTableFlags_BordersOuter | 
                                           ImGuiTableFlags_BordersV;
            
            ImVec2 outer_size = ImVec2(0.0f, ImGui::GetTextLineHeightWithSpacing() * 6);
            if (ImGui::BeginTable("table1", 4, flags, outer_size))
            {
                ImGui::TableSetupColumn("Id");
                ImGui::TableSetupColumn("Uniform Buffer");
                ImGui::TableSetupColumn("Name");
                ImGui::TableSetupColumn("Type");
                ImGui::TableHeadersRow();

                for (size_t j = 0; j < uniforms.size(); j++)
                {
                    ImGui::TableNextRow();

                    ImGui::TableSetColumnIndex(0);
                    ImGui::Text("%d", uniforms[j].mUniformId);
                    ImGui::TableSetColumnIndex(1);
                    ImGui::Text("%s", uniforms[j].mBufferName.c_str());
                    ImGui::TableSetColumnIndex(2);
                    ImGui::Text("%s", uniforms[j].mName.c_str());
                    ImGui::TableSetColumnIndex(3);
                    ImGui::Text("%s", PhysicsEngine::ShaderUniformTypeToString(uniforms[j].mType));
                }

                ImGui::EndTable();
            }
        }

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

        // if selected shader changes, refill buffers
        if (mShaderId != shader->getGuid())
        {
            vertexTextFilled = false;
            fragmentTextFilled = false;
            geometryTextFilled = false;
        }
        mShaderId = shader->getGuid();

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
        else if (geometryShaderActive)
        {
            text = &geometryText[0];
        }

        assert(text != nullptr);

        static ImGuiInputTextFlags flags = ImGuiInputTextFlags_AllowTabInput;
        ImGui::InputTextMultiline("Shader Source", text, 16384, ImVec2(-FLT_MIN, ImGui::GetTextLineHeight() * 32), flags);

        if (ImGui::BeginTable("Compilation Status", 2, ImGuiTableFlags_SizingFixedFit))
        {
            // Determine if shader has been edited
            bool vertexEditMade = (strcmp(shader->getVertexShader().c_str(), &vertexText[0]) != 0);
            bool fragmentEditMade = (strcmp(shader->getFragmentShader().c_str(), &fragmentText[0]) != 0);
            bool geometryEditMade = (strcmp(shader->getGeometryShader().c_str(), &geometryText[0]) != 0);

            // Determine if all shader variants compiled successfully
            char vertexCompileLog[512];
            char fragmentCompileLog[512];
            char geometryCompileLog[512];
            char linkLog[512];
            bool vertexCompiled = true;
            bool fragmentCompiled = true;
            bool geometryCompiled = true;
            bool linked = true;
            for (size_t i = 0; i < programs.size(); i++)
            {
                if (vertexCompiled && !programs[i]->getStatus().mVertexShaderCompiled)
                {
                    strncpy(vertexCompileLog, programs[i]->getStatus().mVertexCompileLog, sizeof(vertexCompileLog));
                    vertexCompiled = false;
                }
                if (fragmentCompiled && !programs[i]->getStatus().mFragmentShaderCompiled)
                {
                    strncpy(fragmentCompileLog, programs[i]->getStatus().mFragmentCompileLog, sizeof(fragmentCompileLog));
                    fragmentCompiled = false;
                }
                if (geometryCompiled && !programs[i]->getStatus().mGeometryShaderCompiled)
                {
                    strncpy(geometryCompileLog, programs[i]->getStatus().mGeometryCompileLog, sizeof(geometryCompileLog));
                    geometryCompiled = false;
                }
                if (linked && !programs[i]->getStatus().mShaderLinked)
                {
                    strncpy(linkLog, programs[i]->getStatus().mLinkLog, sizeof(linkLog));
                    linked = false;
                }
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
                    ImGui::TextColored(ImVec4(1.0f, 0.0f, 0.0f, 1.0f), vertexCompileLog);
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
                    ImGui::TextColored(ImVec4(1.0f, 0.0f, 0.0f, 1.0f), fragmentCompileLog);
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
                    ImGui::TextColored(ImVec4(1.0f, 0.0f, 0.0f, 1.0f), geometryCompileLog);
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
                ImGui::TextColored(ImVec4(1.0f, 0.0f, 0.0f, 1.0f), linkLog);
            }

            ImGui::EndTable();
        }

        //ImGui::Separator();

        ImGui::SetCursorPosX(glm::max(ImGui::GetWindowSize().x - 80.0f, 0.0f));
        if (ImGui::Button("Recompile"))
        {
            shader->setVertexShader(std::string(vertexText));
            shader->setFragmentShader(std::string(fragmentText));
            shader->setGeometryShader(std::string(geometryText));
        }

        ImGui::Separator();


        //if (ImGui::Button("Reload from file"))
        //{
        //    vertexTextFilled = false;
        //    fragmentTextFilled = false;
        //    geometryTextFilled = false;
        //}

        if (ImGui::Button("Save to file"))
        {
            std::ofstream file(shader->getSourceFilepath());
            file << "#vertex\n";
            file << shader->getVertexShader() + "\n";
            file << "#fragment\n";
            file << shader->getFragmentShader() + "\n";
            file.close();
        }
    }

    mContentMax = ImGui::GetItemRectMax();
}

bool ShaderDrawer::isHovered() const
{
    ImVec2 cursorPos = ImGui::GetMousePos();

    glm::vec2 min = glm::vec2(mContentMin.x, mContentMin.y);
    glm::vec2 max = glm::vec2(mContentMax.x, mContentMax.y);

    PhysicsEngine::Rect rect(min, max);

    return rect.contains(cursorPos.x, cursorPos.y);
}