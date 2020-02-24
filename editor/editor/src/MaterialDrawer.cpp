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
	Shader* shader = world->getAsset<Shader>(material->shaderId);

	std::string temp = shader->assetId.toString();
	const char* current_item = temp.c_str();

	if (ImGui::BeginCombo("Shader", current_item, ImGuiComboFlags_None))
	{
		for (int i = 0; i < world->getNumberOfAssets<Shader>(); i++) {
			Shader* s = world->getAssetByIndex<Shader>(i);
			std::string id = s->assetId.toString();
			bool is_selected = (current_item == id);
			if (ImGui::Selectable(id.c_str(), is_selected)) {
				current_item = id.c_str();
				CommandManager::addCommand(new ChangePropertyCommand<Guid>(&material->shaderId, id, &project.isDirty));
			}
			if (is_selected) {
				ImGui::SetItemDefaultFocus();
			}
		}
		ImGui::EndCombo();
	}

    //Shader* shader = world->getAsset<Shader>(material->shaderId);

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
			if (ImGui::InputInt(uniforms[i].shortName, &temp)){
				CommandManager::addCommand(new ChangePropertyCommand<int>(&temp, temp, &project.isDirty));
			}
		}
		else if (uniforms[i].type == GL_FLOAT) {
			float temp = shader->getFloat(uniforms[i].name);
			//Log::info(std::to_string(temp).c_str());
			if (ImGui::InputFloat(uniforms[i].shortName, &temp))
			{
				CommandManager::addCommand(new ChangePropertyCommand<float>(&temp, temp, &project.isDirty));
			}
		}
		else if (uniforms[i].type == GL_FLOAT_VEC2) {
			glm::vec2 temp = glm::vec2(0.0f);
			if (ImGui::InputFloat2(uniforms[i].shortName, &temp[0])) 
			{
				CommandManager::addCommand(new ChangePropertyCommand<glm::vec2>(&temp, temp, &project.isDirty));
			}
		}
		else if (uniforms[i].type == GL_FLOAT_VEC3) {
			glm::vec3 temp = glm::vec3(0.0f);
			if (ImGui::InputFloat3(uniforms[i].shortName, &temp[0]))
			{
				CommandManager::addCommand(new ChangePropertyCommand<glm::vec3>(&temp, temp, &project.isDirty));
			}
		}
		else if (uniforms[i].type == GL_FLOAT_VEC4) {
			glm::vec4 temp = glm::vec4(0.0f);
			if (ImGui::InputFloat4(uniforms[i].shortName, &temp[0]))
			{
				CommandManager::addCommand(new ChangePropertyCommand<glm::vec4>(&temp, temp, &project.isDirty));
			}
		}

		if (uniforms[i].type == GL_SAMPLER_2D) {
		
			Guid textureId = material->getTexture(uniforms[i].name);
			std::string test = textureId.toString();

			Texture2D* texture = world->getAsset<Texture2D>(textureId);

			std::string temp1 = textureId.toString();

			bool slotFilled = false;
			bool isClicked = ImGui::ImageSlot(uniforms[i].shortName, texture == NULL ? 0 : texture->handle.handle, clipboard.getDraggedType() == InteractionType::Texture2D, &slotFilled);
			if (slotFilled) {
				textureId = clipboard.getDraggedId();
				clipboard.clearDraggedItem();

				std::string temp2 = textureId.toString();

				material->setTexture(uniforms[i].name, textureId);

				project.isDirty = true;
				//CommandManager::addCommand(new ChangePropertyCommand<int>(&temp, temp, &project.isDirty));
			}
		}
    }

}