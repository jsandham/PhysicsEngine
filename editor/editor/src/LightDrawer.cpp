#include "../include/LightDrawer.h"
#include "../include/CommandManager.h"
#include "../include/EditorCommands.h"

#include "components/Light.h"

#include "../include/imgui/imgui.h"
#include "../include/imgui/imgui_impl_win32.h"
#include "../include/imgui/imgui_impl_opengl3.h"
#include "../include/imgui/imgui_internal.h"

using namespace PhysicsEditor;

LightDrawer::LightDrawer()
{

}

LightDrawer::~LightDrawer()
{

}

void LightDrawer::render(World* world, EditorClipboard& clipboard, Guid id)
{
	if(ImGui::TreeNodeEx("Light", ImGuiTreeNodeFlags_DefaultOpen))
	{
		Light* light = world->getComponentById<Light>(id);

		ImGui::Text(("EntityId: " + light->entityId.toString()).c_str());
		ImGui::Text(("ComponentId: " + id.toString()).c_str());

		glm::vec3 position = light->position;
		glm::vec3 direction = light->direction;
		glm::vec3 ambient = light->ambient;
		glm::vec3 diffuse = light->diffuse;
		glm::vec3 specular = light->specular;

		if (ImGui::InputFloat3("Position", glm::value_ptr(position))) {
			CommandManager::addCommand(new ChangePropertyCommand<glm::vec3>(&light->position, position));
		}
		if (ImGui::InputFloat3("Direction", glm::value_ptr(direction))) {
			CommandManager::addCommand(new ChangePropertyCommand<glm::vec3>(&light->direction, direction));
		}
		if (ImGui::InputFloat3("Ambient", glm::value_ptr(ambient))) {
			CommandManager::addCommand(new ChangePropertyCommand<glm::vec3>(&light->ambient, ambient));
		}
		if (ImGui::InputFloat3("Diffuse", glm::value_ptr(diffuse))) {
			CommandManager::addCommand(new ChangePropertyCommand<glm::vec3>(&light->diffuse, diffuse));
		}
		if (ImGui::InputFloat3("Specular", glm::value_ptr(specular))) {
			CommandManager::addCommand(new ChangePropertyCommand<glm::vec3>(&light->specular, specular));
		}

		float constant = light->constant;
		float linear = light->linear;
		float quadratic = light->quadratic;
		float cutOff = light->cutOff;
		float outerCutOff = light->outerCutOff;

		if (ImGui::InputFloat("Constant", &constant)) {
			CommandManager::addCommand(new ChangePropertyCommand<float>(&light->constant, constant));
		}
		if (ImGui::InputFloat("Linear", &linear)) {
			CommandManager::addCommand(new ChangePropertyCommand<float>(&light->linear, linear));
		}
		if (ImGui::InputFloat("Quadratic", &quadratic)) {
			CommandManager::addCommand(new ChangePropertyCommand<float>(&light->quadratic, quadratic));
		}
		if (ImGui::InputFloat("Cut-Off", &cutOff)) {
			CommandManager::addCommand(new ChangePropertyCommand<float>(&light->cutOff, cutOff));
		}
		if (ImGui::InputFloat("Outer Cut-Off", &outerCutOff)) {
			CommandManager::addCommand(new ChangePropertyCommand<float>(&light->outerCutOff, outerCutOff));
		}

		const char* lightTypes[] = { "Directional", "Spot", "Point" };
		int lightTypeIndex = static_cast<int>(light->lightType);
		if (ImGui::Combo("##LightType", &lightTypeIndex, lightTypes, IM_ARRAYSIZE(lightTypes))) {
			CommandManager::addCommand(new ChangePropertyCommand<LightType>(&light->lightType, static_cast<LightType>(lightTypeIndex)));
		}
		
		const char* shadowTypes[] = { "Hard", "Soft" };
		int shadowTypeIndex = static_cast<int>(light->shadowType);
		if (ImGui::Combo("##ShadowType", &shadowTypeIndex, shadowTypes, IM_ARRAYSIZE(shadowTypes))) {
			CommandManager::addCommand(new ChangePropertyCommand<ShadowType>(&light->shadowType, static_cast<ShadowType>(shadowTypeIndex)));
		}

		ImGui::TreePop();
	}
}