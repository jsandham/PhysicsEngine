#include "../include/MaterialDrawer.h"
#include "../include/CommandManager.h"
#include "../include/EditorCommands.h"

#include "core/Mesh.h"
#include "core/InternalMeshes.h"

#include "systems/RenderSystem.h"
#include "systems/CleanUpSystem.h"

using namespace PhysicsEditor;

MaterialDrawer::MaterialDrawer()
{
	materialViewWorldPopulated = false;
}

MaterialDrawer::~MaterialDrawer()
{

}

void MaterialDrawer::render(World* world, EditorProject& project, EditorScene& scene, EditorClipboard& clipboard, Guid id)
{
	Material* material = world->getAsset<Material>(id);
	Shader* shader = world->getAsset<Shader>(material->getShaderId());

	Guid currentShaderId = material->getShaderId();

	// dropdown for selecting shader for material
	if (ImGui::BeginCombo("Shader", currentShaderId.toString().c_str(), ImGuiComboFlags_None))
	{
		for (int i = 0; i < world->getNumberOfAssets<Shader>(); i++) {
			Shader* s = world->getAssetByIndex<Shader>(i);
			
			bool is_selected = (currentShaderId == s->getId());
			if (ImGui::Selectable(s->getId().toString().c_str(), is_selected)) {
				currentShaderId = s->getId();

				material->setShaderId(currentShaderId);
				material->onShaderChanged(world);
			}
			if (is_selected) {
				ImGui::SetItemDefaultFocus();
			}
		}
		ImGui::EndCombo();
	}

	// draw material uniforms
	std::vector<ShaderUniform> uniforms = material->getUniforms();
    for(size_t i = 0; i < uniforms.size(); i++)
	{
		// only expose uniforms exist in a Material uniform struct in the shader
		if (std::strcmp(uniforms[i].mBlockName, "material") != 0)
		{
			continue;
		}

		// Note: matrices not supported
		switch (uniforms[i].mType)
		{
			case GL_INT:
				UniformDrawer<GL_INT>::draw(world, material, &uniforms[i], clipboard, project);
				break;
			case GL_FLOAT:
				UniformDrawer<GL_FLOAT>::draw(world, material, &uniforms[i], clipboard, project);
				break;
			case GL_FLOAT_VEC2:
				UniformDrawer<GL_FLOAT_VEC2>::draw(world, material, &uniforms[i], clipboard, project);
				break;
			case GL_FLOAT_VEC3:
				UniformDrawer<GL_FLOAT_VEC3>::draw(world, material, &uniforms[i], clipboard, project);
				break;
			case GL_FLOAT_VEC4:
				UniformDrawer<GL_FLOAT_VEC4>::draw(world, material, &uniforms[i], clipboard, project);
				break;
			case GL_SAMPLER_2D:
				UniformDrawer<GL_SAMPLER_2D>::draw(world, material, &uniforms[i], clipboard, project);
				break;
			case GL_SAMPLER_CUBE:
				UniformDrawer<GL_SAMPLER_CUBE>::draw(world, material, &uniforms[i], clipboard, project);
				break;

		}
    }

	ImGui::Separator();

	ImGui::Text("Preview");

	// Draw material preview child window
	{
		if (!materialViewWorldPopulated) {
			populateMaterialViewWorld(material, shader);

			int t = materialViewWorld.getNumberOfSystems();

			for (int i = 0; i < materialViewWorld.getNumberOfSystems(); i++) {
				System* system = materialViewWorld.getSystemByIndex(i);

				system->init(&materialViewWorld);
			}

			materialViewWorldPopulated = true;
		}

		for (int i = 0; i < materialViewWorld.getNumberOfSystems(); i++) {
			System* system = materialViewWorld.getSystemByIndex(i);

			system->update({});
		}

		Log::info("\n");
		Log::info(("material view world system count: " + std::to_string(materialViewWorld.getNumberOfSystems()) + "\n").c_str());
		Log::info(("material view world entity count: " + std::to_string(materialViewWorld.getNumberOfEntities()) + "\n").c_str());
		Log::info(("material view world material count: " + std::to_string(materialViewWorld.getNumberOfAssets<Material>()) + "\n").c_str());
		Log::info(("material view world mesh count: " + std::to_string(materialViewWorld.getNumberOfAssets<Mesh>()) + "\n").c_str());
		Log::info(("material view world shader count: " + std::to_string(materialViewWorld.getNumberOfAssets<Shader>()) + "\n").c_str());
		Log::info(("material view world texture count: " + std::to_string(materialViewWorld.getNumberOfAssets<Texture2D>()) + "\n").c_str());

		RenderSystem* worldRS = world->getFirstSystem<RenderSystem>();
		RenderSystem* renderSystem = materialViewWorld.getFirstSystem<RenderSystem>();

		std::string test1 = worldRS->getId().toString();
		std::string test2 = renderSystem->getId().toString();

		GraphicsTargets targets = renderSystem->getGraphicsTargets();

		ImGuiWindowFlags window_flags = ImGuiWindowFlags_None;// ImGuiWindowFlags_HorizontalScrollbar | (disable_mouse_wheel ? ImGuiWindowFlags_NoScrollWithMouse : 0);
		ImGui::BeginChild("MaterialPreviewWindow", ImVec2(ImGui::GetWindowContentRegionWidth(), ImGui::GetWindowContentRegionWidth()), true, window_flags);

		ImGui::Image((void*)(intptr_t)targets.mColor, ImVec2(ImGui::GetWindowContentRegionWidth(), ImGui::GetWindowContentRegionWidth()), ImVec2(1, 1), ImVec2(0, 0));

		ImGui::EndChild();
	}
}

void MaterialDrawer::populateMaterialViewWorld(Material* material, Shader* shader)
{
	// create shader
	Shader* sphereShader = materialViewWorld.createAsset<Shader>();
	sphereShader->load(shader->getVertexShader(), shader->getFragmentShader(), shader->getGeometryShader());

	std::string test = sphereShader->getId().toString();

	// create material
	Material* sphereMaterial = materialViewWorld.createAsset<Material>();
	sphereMaterial->setShaderId(sphereShader->getId());
	sphereMaterial->onShaderChanged(&materialViewWorld);

	// create sphere mesh
	Mesh* sphereMesh = materialViewWorld.createAsset<Mesh>();
	sphereMesh->load(PhysicsEngine::InternalMeshes::sphereVertices,
					 PhysicsEngine::InternalMeshes::sphereNormals,
					 PhysicsEngine::InternalMeshes::sphereTexCoords,
					 PhysicsEngine::InternalMeshes::sphereSubMeshStartIndicies);

	// create sphere entity in world 
	Entity* sphereEntity = materialViewWorld.createEntity();

	Transform* transform = sphereEntity->addComponent<Transform>(&materialViewWorld);
	transform->mPosition = glm::vec3(0.0f, 0.0f, 0.0f);
	transform->mRotation = glm::quat(0.0f, 0.0f, 0.0f, 1.0f);
	transform->mScale = glm::vec3(1.0f, 1.0f, 1.0f);

	MeshRenderer* meshRenderer = sphereEntity->addComponent<MeshRenderer>(&materialViewWorld);
	meshRenderer->mMeshId = sphereMesh->getId();
	meshRenderer->mMaterialIds[0] = sphereMaterial->getId();
	meshRenderer->mMaterialCount = 1;
	meshRenderer->mIsStatic = false;

	// create light entity in world
	Entity* lightEntity = materialViewWorld.createEntity();

	transform = lightEntity->addComponent<Transform>(&materialViewWorld);
	transform->mPosition = glm::vec3(0.0f, 0.0f, 0.0f);
	transform->mRotation = glm::quat(0.0f, 0.0f, 0.0f, 1.0f);
	transform->mScale = glm::vec3(1.0f, 1.0f, 1.0f);

	Light* light = lightEntity->addComponent<Light>(&materialViewWorld);

	// create camera entity in world
	Entity* cameraEntity = materialViewWorld.createEntity();

	transform = cameraEntity->addComponent<Transform>(&materialViewWorld);
	transform->mPosition = glm::vec3(0.0f, 0.0f, 0.0f);
	transform->mRotation = glm::quat(0.0f, 0.0f, 0.0f, 1.0f);
	transform->mScale = glm::vec3(1.0f, 1.0f, 1.0f);

	Camera* camera = cameraEntity->addComponent<Camera>(&materialViewWorld);
	camera->mPosition = glm::vec3(5.0f, 0.0f, 0.0f);
	camera->mFront = glm::vec3(-5.0f, 0.0f, 0.0f);
	camera->mUp = glm::vec3(0.0f, 0.0f, 1.0f);

	// add render system to material view world
	RenderSystem* renderSystem = materialViewWorld.addSystem<RenderSystem>(0);
	renderSystem->mRenderToScreen = false;

	// add required clean-up system to material view world
	CleanUpSystem* cleanUpSystem = materialViewWorld.addSystem<CleanUpSystem>(1);
}