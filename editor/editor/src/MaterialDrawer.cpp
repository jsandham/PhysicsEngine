#include "../include/MaterialDrawer.h"
#include "../include/CommandManager.h"
#include "../include/EditorCommands.h"

#include "core/Material.h"

#include "../include/imgui/imgui.h"
#include "../include/imgui/imgui_impl_win32.h"
#include "../include/imgui/imgui_impl_opengl3.h"
#include "../include/imgui/imgui_internal.h"
#include "../include/imgui_extensions.h"

using namespace PhysicsEditor;

MaterialDrawer::MaterialDrawer()
{

}

MaterialDrawer::~MaterialDrawer()
{

}

void MaterialDrawer::render(World* world, EditorProject& project, EditorScene& scene, EditorClipboard& clipboard, Guid id)
{
	Material* material = world->getAsset<Material>(id);

	int currentShaderIndex = -1;
	std::vector<std::string> shaderNames;
	for (int i = 0; i < world->getNumberOfAssets<Shader>(); i++) {
		Shader* shader = world->getAssetByIndex<Shader>(i);
		shaderNames.push_back(shader->assetId.toString());

		if (material->shaderId	== shader->assetId) {
			currentShaderIndex = i;
		}
	}

	if (ImGui::Combo("Shader", &currentShaderIndex, shaderNames)) {
		CommandManager::addCommand(new ChangePropertyCommand<Guid>(&material->shaderId, world->getAssetByIndex<Shader>(currentShaderIndex)->assetId, &project.isDirty));
	}

    Shader* shader = world->getAsset<Shader>(material->shaderId);

    std::vector<ShaderUniform> uniforms = shader->getUniforms();
    for(size_t i = 0; i < uniforms.size(); i++){
		//std::string name = std::string(uniforms[i].name);
		//size_t startIndex = name.find_last_of(".") + 1;
		//std::string labelName = name.substr(startIndex, name.length() - startIndex);

		// Note: matrices not supported
		/*switch (uniforms[i].type)
		{
			case GL_INT:
				UniformDrawer<GL_INT>::draw();
				break;
			case GL_INT_VEC2:
				UniformDrawer<GL_INT_VEC2>::draw();
				break;
			case GL_INT_VEC3:
				UniformDrawer<GL_INT_VEC3>::draw();
				break;
			case GL_INT_VEC4:
				UniformDrawer<GL_INT_VEC4>::draw();
				break;
			case GL_FLOAT:
				UniformDrawer<GL_FLOAT>::draw();
				break;
			case GL_FLOAT_VEC2:
				UniformDrawer<GL_FLOAT_VEC2>::draw();
				break;
			case GL_FLOAT_VEC3:
				UniformDrawer<GL_FLOAT_VEC3>::draw();
				break;
			case GL_FLOAT_VEC4:
				UniformDrawer<GL_FLOAT_VEC4>::draw();
				break;
			case GL_SAMPLER_2D:
				UniformDrawer<GL_SAMPLER_2D>::draw();
				break;
			case GL_SAMPLER_CUBE:
				UniformDrawer<GL_SAMPLER_CUBE>::draw();
				break;
			default:

		}*/

		if (uniforms[i].type == GL_INT) {
			int temp = shader->getInt(uniforms[i].name);
			if (ImGui::InputInt(uniforms[i].shortName.c_str(), &temp)){
				CommandManager::addCommand(new ChangePropertyCommand<int>(&temp, temp, &project.isDirty));
			}
		}
		else if (uniforms[i].type == GL_FLOAT) {
			float temp = shader->getFloat(uniforms[i].name);
			//Log::info(std::to_string(temp).c_str());
			if (ImGui::InputFloat(uniforms[i].shortName.c_str(), &temp))
			{
				CommandManager::addCommand(new ChangePropertyCommand<float>(&temp, temp, &project.isDirty));
			}
		}
		else if (uniforms[i].type == GL_FLOAT_VEC2) {
			glm::vec2 temp = glm::vec2(0.0f);
			if (ImGui::InputFloat2(uniforms[i].shortName.c_str(), &temp[0])) 
			{
				CommandManager::addCommand(new ChangePropertyCommand<glm::vec2>(&temp, temp, &project.isDirty));
			}
		}
		else if (uniforms[i].type == GL_FLOAT_VEC3) {
			glm::vec3 temp = glm::vec3(0.0f);
			if (ImGui::InputFloat3(uniforms[i].shortName.c_str(), &temp[0]))
			{
				CommandManager::addCommand(new ChangePropertyCommand<glm::vec3>(&temp, temp, &project.isDirty));
			}
		}
		else if (uniforms[i].type == GL_FLOAT_VEC4) {
			glm::vec4 temp = glm::vec4(0.0f);
			if (ImGui::InputFloat4(uniforms[i].shortName.c_str(), &temp[0]))
			{
				CommandManager::addCommand(new ChangePropertyCommand<glm::vec4>(&temp, temp, &project.isDirty));
			}
		}
    }

}