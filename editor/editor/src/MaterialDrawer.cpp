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

void MaterialDrawer::render(World* world, EditorClipboard& clipboard, Guid id)
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
		CommandManager::addCommand(new ChangePropertyCommand<Guid>(&material->shaderId, world->getAssetByIndex<Shader>(currentShaderIndex)->assetId));
	}

    Shader* shader = world->getAsset<Shader>(material->shaderId);

    std::vector<ShaderUniform> uniforms = shader->getUniforms();
    for(size_t i = 0; i < uniforms.size(); i++){
		std::string name = std::string(uniforms[i].name);
		size_t startIndex = name.find_last_of(".") + 1;
		std::string labelName = name.substr(startIndex, name.length() - startIndex);

		if (uniforms[i].type == ShaderDataType::GLIntVec1) {
			int temp = 0;
			if (ImGui::InputInt(labelName.c_str(), &temp)){
				CommandManager::addCommand(new ChangePropertyCommand<int>(&temp, temp));
			}
		}
		else if (uniforms[i].type == ShaderDataType::GLFloatVec1) {
			float temp = 0.0f;
			if (ImGui::InputFloat(labelName.c_str(), &temp))
			{
				CommandManager::addCommand(new ChangePropertyCommand<float>(&temp, temp));
			}
		}
		else if (uniforms[i].type == ShaderDataType::GLFloatVec2) {
			glm::vec2 temp = glm::vec2(0.0f);
			if (ImGui::InputFloat2(labelName.c_str(), &temp[0])) 
			{
				CommandManager::addCommand(new ChangePropertyCommand<glm::vec2>(&temp, temp));
			}
		}
		else if (uniforms[i].type == ShaderDataType::GLFloatVec3) {
			glm::vec3 temp = glm::vec3(0.0f);
			if (ImGui::InputFloat3(labelName.c_str(), &temp[0]))
			{
				CommandManager::addCommand(new ChangePropertyCommand<glm::vec3>(&temp, temp));
			}
		}
		else if (uniforms[i].type == ShaderDataType::GLFloatVec4) {
			glm::vec4 temp = glm::vec4(0.0f);
			if (ImGui::InputFloat4(labelName.c_str(), &temp[0]))
			{
				CommandManager::addCommand(new ChangePropertyCommand<glm::vec4>(&temp, temp));
			}
		}
		else if (uniforms[i].type == ShaderDataType::GLFloatMat2) {
		
		}
		else if (uniforms[i].type == ShaderDataType::GLFloatMat3) {

		}
		else if (uniforms[i].type == ShaderDataType::GLFloatMat4) {

		}
    }

}