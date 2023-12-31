#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <TextEditor.h>

#include "../../include/drawers/ShaderDrawer.h"
#include "../../include/imgui/imgui_extensions.h"

#include "core/Shader.h"
#include "core/Log.h"
#include "graphics/RenderContext.h"

using namespace PhysicsEditor;

ShaderDrawer::ShaderDrawer()
{
	mShaderId = PhysicsEngine::Guid::INVALID;
}

ShaderDrawer::~ShaderDrawer()
{
}

void ShaderDrawer::render(Clipboard& clipboard, const PhysicsEngine::Guid& id)
{
	ImGui::Separator();
	mContentMin = ImGui::GetItemRectMin();

	PhysicsEngine::Shader* shader = clipboard.getWorld()->getAssetByGuid<PhysicsEngine::Shader>(id);

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
			ImGui::Text(shader->mName.c_str());

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

		if (ImGui::Begin("Shader Editor", NULL))
		{
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

			static std::string vertexText;
			static std::string geometryText;
			static std::string fragmentText;

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
				vertexText = shader->getVertexShader();
				vertexTextFilled = true;
			}
			if (!fragmentTextFilled)
			{
				fragmentText = shader->getFragmentShader();
				fragmentTextFilled = true;
			}
			if (!geometryTextFilled)
			{
				geometryText = shader->getGeometryShader();
				geometryTextFilled = true;
			}

			static std::string text;
			if (vertexShaderActive)
			{
				text = vertexText;
			}
			else if (fragmentShaderActive)
			{
				text = fragmentText;
			}
			else if (geometryShaderActive)
			{
				text = geometryText;
			}

			static TextEditor mEditor;
			mEditor.SetLanguageDefinition(TextEditor::LanguageDefinitionId::Cpp);
			mEditor.SetText(text);
			mEditor.Render("ShaderTextEditor", false, ImVec2(-FLT_MIN, ImGui::GetTextLineHeight() * 32), true);

			if (vertexShaderActive)
			{
				vertexText = mEditor.GetText();
			}
			else if (fragmentShaderActive)
			{
				fragmentText = mEditor.GetText();
			}
			else if (geometryShaderActive)
			{
				geometryText = mEditor.GetText();
			}

			if (ImGui::BeginTable("Compilation Status", 2, ImGuiTableFlags_SizingFixedFit))
			{
				// Determine if shader has been edited
				bool vertexEditMade = shader->getVertexShader().compare(vertexText) != 0;
				bool fragmentEditMade = shader->getFragmentShader().compare(fragmentText) != 0;
				bool geometryEditMade = shader->getGeometryShader().compare(geometryText) != 0;

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


			if (ImGui::Button("Reload from file"))
			{
			    vertexTextFilled = false;
			    fragmentTextFilled = false;
			    geometryTextFilled = false;
			}

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

		ImGui::End();
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