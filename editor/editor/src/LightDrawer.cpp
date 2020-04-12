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

void LightDrawer::render(World* world, EditorProject& project, EditorScene& scene, EditorClipboard& clipboard, Guid id)
{
	if(ImGui::TreeNodeEx("Light", ImGuiTreeNodeFlags_DefaultOpen))
	{
		Light* light = world->getComponentById<Light>(id);

		ImGui::Text(("EntityId: " + light->getEntityId().toString()).c_str());
		ImGui::Text(("ComponentId: " + id.toString()).c_str());

		glm::vec3 ambient = light->mAmbient;
		glm::vec3 diffuse = light->mDiffuse;
		glm::vec3 specular = light->mSpecular;

		if (ImGui::InputFloat3("Ambient", glm::value_ptr(ambient))) {
			CommandManager::addCommand(new ChangePropertyCommand<glm::vec3>(&light->mAmbient, ambient, &scene.isDirty));
		}
		if (ImGui::InputFloat3("Diffuse", glm::value_ptr(diffuse))) {
			CommandManager::addCommand(new ChangePropertyCommand<glm::vec3>(&light->mDiffuse, diffuse, &scene.isDirty));
		}
		if (ImGui::InputFloat3("Specular", glm::value_ptr(specular))) {
			CommandManager::addCommand(new ChangePropertyCommand<glm::vec3>(&light->mSpecular, specular, &scene.isDirty));
		}

		float constant = light->mConstant;
		float linear = light->mLinear;
		float quadratic = light->mQuadratic;
		float cutOff = light->mCutOff;
		float outerCutOff = light->mOuterCutOff;

		if (ImGui::InputFloat("Constant", &constant)) {
			CommandManager::addCommand(new ChangePropertyCommand<float>(&light->mConstant, constant, &scene.isDirty));
		}
		if (ImGui::InputFloat("Linear", &linear)) {
			CommandManager::addCommand(new ChangePropertyCommand<float>(&light->mLinear, linear, &scene.isDirty));
		}
		if (ImGui::InputFloat("Quadratic", &quadratic)) {
			CommandManager::addCommand(new ChangePropertyCommand<float>(&light->mQuadratic, quadratic, &scene.isDirty));
		}
		if (ImGui::InputFloat("Cut-Off", &cutOff)) {
			CommandManager::addCommand(new ChangePropertyCommand<float>(&light->mCutOff, cutOff, &scene.isDirty));
		}
		if (ImGui::InputFloat("Outer Cut-Off", &outerCutOff)) {
			CommandManager::addCommand(new ChangePropertyCommand<float>(&light->mOuterCutOff, outerCutOff, &scene.isDirty));
		}

		const char* lightTypes[] = { "Directional", "Spot", "Point" };
		int lightTypeIndex = static_cast<int>(light->mLightType);
		if (ImGui::Combo("##LightType", &lightTypeIndex, lightTypes, IM_ARRAYSIZE(lightTypes))) {
			CommandManager::addCommand(new ChangePropertyCommand<LightType>(&light->mLightType, static_cast<LightType>(lightTypeIndex), &scene.isDirty));
		}
		
		const char* shadowTypes[] = { "Hard", "Soft" };
		int shadowTypeIndex = static_cast<int>(light->mShadowType);
		if (ImGui::Combo("##ShadowType", &shadowTypeIndex, shadowTypes, IM_ARRAYSIZE(shadowTypes))) {
			CommandManager::addCommand(new ChangePropertyCommand<ShadowType>(&light->mShadowType, static_cast<ShadowType>(shadowTypeIndex), &scene.isDirty));
		}

		ImGui::TreePop();
	}
}