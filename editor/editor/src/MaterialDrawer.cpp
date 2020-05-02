#include "../include/MaterialDrawer.h"
#include "../include/CommandManager.h"
#include "../include/EditorCommands.h"

#include "core/Mesh.h"
#include "core/InternalMeshes.h"
#include "core/MaterialUtil.h"

#include "systems/RenderSystem.h"
#include "systems/CleanUpSystem.h"

using namespace PhysicsEditor;

MaterialDrawer::MaterialDrawer()
{
	//initViewWorld();
}

MaterialDrawer::~MaterialDrawer()
{

}

void MaterialDrawer::render(World* world, EditorProject& project, EditorScene& scene, EditorClipboard& clipboard, Guid id)
{
	Material* material = world->getAsset<Material>(id);

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

	// Draw material preview child window
	ImGui::Text("Preview");

	if (previewWorld.getNumberOfEntities() == 0) {
		populatePreviewWorld(world);

		for (int i = 0; i < previewWorld.getNumberOfSystems(); i++) {
			System* system = previewWorld.getSystemByIndex(i);

			system->init(&previewWorld);
		}
	}

	// copy currently selected material from world to corresponding material in preview world
	Material* previewMaterial = previewWorld.getAsset<Material>(material->getId());
	MaterialUtil::copyMaterialTo(world, material, &previewWorld, previewMaterial);

	// set preview material on sphere meshrenderer
	sphereMeshRenderer->setMaterial(previewMaterial->getId(), 0);

	//Log::info(("material view world material count: " + std::to_string(previewWorld.getNumberOfAssets<Material>()) + "\n").c_str());
	//Log::info(("material view world mesh count: " + std::to_string(previewWorld.getNumberOfAssets<Mesh>()) + "\n").c_str());
	//Log::info(("material view world shader count: " + std::to_string(previewWorld.getNumberOfAssets<Shader>()) + "\n").c_str());
	//Log::info(("material view world texture count: " + std::to_string(previewWorld.getNumberOfAssets<Texture2D>()) + "\n").c_str());

	for (int i = 0; i < previewWorld.getNumberOfSystems(); i++) {
		System* system = previewWorld.getSystemByIndex(i);

		system->update({}, {});
	}

	//Log::info("\n");
	//Log::info(("material view world system count: " + std::to_string(previewWorld.getNumberOfSystems()) + "\n").c_str());
	//Log::info(("material view world entity count: " + std::to_string(previewWorld.getNumberOfEntities()) + "\n").c_str());
	//Log::info(("material view world material count: " + std::to_string(previewWorld.getNumberOfAssets<Material>()) + "\n").c_str());
	//Log::info(("material view world mesh count: " + std::to_string(previewWorld.getNumberOfAssets<Mesh>()) + "\n").c_str());
	//Log::info(("material view world shader count: " + std::to_string(previewWorld.getNumberOfAssets<Shader>()) + "\n").c_str());
	//Log::info(("material view world texture count: " + std::to_string(previewWorld.getNumberOfAssets<Texture2D>()) + "\n").c_str());

	RenderSystem* renderSystem = previewWorld.getSystem<RenderSystem>();

	GraphicsTargets targets = renderSystem->getGraphicsTargets();

	ImGuiWindowFlags window_flags = ImGuiWindowFlags_None;// ImGuiWindowFlags_HorizontalScrollbar | (disable_mouse_wheel ? ImGuiWindowFlags_NoScrollWithMouse : 0);
	ImGui::BeginChild("MaterialPreviewWindow", ImVec2(ImGui::GetWindowContentRegionWidth(), ImGui::GetWindowContentRegionWidth()), true, window_flags);
	ImGui::Image((void*)(intptr_t)targets.mColor, ImVec2(ImGui::GetWindowContentRegionWidth(), ImGui::GetWindowContentRegionWidth()), ImVec2(1, 1), ImVec2(0, 0));
	ImGui::EndChild();
}

void MaterialDrawer::populatePreviewWorld(World* world)
{
	// add all shaders found in world to preview world
	for (int i = 0; i < world->getNumberOfAssets<Shader>(); i++) {
		Shader* shader = world->getAssetByIndex<Shader>(i);

		std::string filepath = world->getAssetFilepath(shader->getId());

		if (!previewWorld.loadAsset(filepath))
		{
			Log::error("Error: Could not load shader into preview world\n");
		}
	}

	// add all materials found in world to preview world
	for (int i = 0; i < world->getNumberOfAssets<Material>(); i++) {
		Material* material = world->getAssetByIndex<Material>(i);

		std::string filepath = world->getAssetFilepath(material->getId());

		if (!previewWorld.loadAsset(filepath))
		{
			Log::error("Error: Could not load material into preview world\n");
		}
	}

	// add all textures found in world to preview world
	for (int i = 0; i < world->getNumberOfAssets<Texture2D>(); i++) {
		Texture2D* texture = world->getAssetByIndex<Texture2D>(i);

		std::string filepath = world->getAssetFilepath(texture->getId());

		if (!previewWorld.loadAsset(filepath))
		{
			Log::error("Error: Could not load texture into preview world\n");
		}
	}

	// create sphere mesh in material preview world
	Mesh* sphereMesh = previewWorld.createAsset<Mesh>();
	sphereMesh->load(PhysicsEngine::InternalMeshes::sphereVertices,
					 PhysicsEngine::InternalMeshes::sphereNormals,
					 PhysicsEngine::InternalMeshes::sphereTexCoords,
					 PhysicsEngine::InternalMeshes::sphereSubMeshStartIndicies);

	// create sphere entity in material view world
	Entity* sphereEntity = previewWorld.createEntity();

	Transform* sphereTransform = sphereEntity->addComponent<Transform>(&previewWorld);
	sphereTransform->mPosition = glm::vec3(0.0f, 0.0f, 0.0f);
	sphereTransform->mRotation = glm::quat(0.0f, 0.0f, 0.0f, 1.0f);
	sphereTransform->mScale = glm::vec3(1.0f, 1.0f, 1.0f);

	sphereMeshRenderer = sphereEntity->addComponent<MeshRenderer>(&previewWorld);
	sphereMeshRenderer->setMesh(sphereMesh->getId());
	sphereMeshRenderer->setMaterial(Guid::INVALID, 0);
	sphereMeshRenderer->mMaterialCount = 1;
	sphereMeshRenderer->mIsStatic = false;

	// create light entity in material view world
	Entity* lightEntity = previewWorld.createEntity();

	Transform* lightTransform = lightEntity->addComponent<Transform>(&previewWorld);
	lightTransform->mPosition = glm::vec3(0.0f, 0.0f, 0.0f);
	lightTransform->mRotation = glm::quat(0.0f, 0.0f, 0.0f, 1.0f);
	lightTransform->mScale = glm::vec3(1.0f, 1.0f, 1.0f);

	Light* light = lightEntity->addComponent<Light>(&previewWorld);

	// create camera entity in material view world
	Entity* cameraEntity = previewWorld.createEntity();

	Transform* cameraTransform = cameraEntity->addComponent<Transform>(&previewWorld);
	cameraTransform->mPosition = glm::vec3(3.0f, 0.0f, 0.0f);
	cameraTransform->mRotation = glm::quat(0.0f, 0.0f, 0.0f, 1.0f);
	cameraTransform->mScale = glm::vec3(1.0f, 1.0f, 1.0f);

	Camera* camera = cameraEntity->addComponent<Camera>(&previewWorld);
	/*camera->mPosition = glm::vec3(3.0f, 0.0f, 0.0f);
	camera->mFront = glm::vec3(-3.0f, 0.0f, 0.0f);
	camera->mUp = glm::vec3(0.0f, 0.0f, 1.0f);*/

	// add render system to material view world
	RenderSystem* renderSystem = previewWorld.addSystem<RenderSystem>(0);
	renderSystem->mRenderToScreen = false;

	// add required clean-up system to material view world
	CleanUpSystem* cleanUpSystem = previewWorld.addSystem<CleanUpSystem>(1);










	//// create material in material view world
	//previewMaterial = previewWorld.createAsset<Material>();
	//previewMaterial->setShaderId(previewShader->getId());

	//// create sphere mesh in material view world
	//Mesh* sphereMesh = previewWorld.createAsset<Mesh>();
	//sphereMesh->load(PhysicsEngine::InternalMeshes::sphereVertices,
	//	PhysicsEngine::InternalMeshes::sphereNormals,
	//	PhysicsEngine::InternalMeshes::sphereTexCoords,
	//	PhysicsEngine::InternalMeshes::sphereSubMeshStartIndicies);

	//// create sphere entity in material view world
	//Entity* sphereEntity = previewWorld.createEntity();

	//Transform* transform = sphereEntity->addComponent<Transform>(&previewWorld);
	//transform->mPosition = glm::vec3(0.0f, 0.0f, 0.0f);
	//transform->mRotation = glm::quat(0.0f, 0.0f, 0.0f, 1.0f);
	//transform->mScale = glm::vec3(1.0f, 1.0f, 1.0f);

	//MeshRenderer* meshRenderer = sphereEntity->addComponent<MeshRenderer>(&previewWorld);
	//meshRenderer->mMeshId = sphereMesh->getId();
	//meshRenderer->mMaterialIds[0] = previewMaterial->getId();
	//meshRenderer->mMaterialCount = 1;
	//meshRenderer->mIsStatic = false;

	//// create light entity in material view world
	//Entity* lightEntity = previewWorld.createEntity();

	//transform = lightEntity->addComponent<Transform>(&previewWorld);
	//transform->mPosition = glm::vec3(0.0f, 0.0f, 0.0f);
	//transform->mRotation = glm::quat(0.0f, 0.0f, 0.0f, 1.0f);
	//transform->mScale = glm::vec3(1.0f, 1.0f, 1.0f);

	//Light* light = lightEntity->addComponent<Light>(&previewWorld);

	//// create camera entity in material view world
	//Entity* cameraEntity = previewWorld.createEntity();

	//transform = cameraEntity->addComponent<Transform>(&previewWorld);
	//transform->mPosition = glm::vec3(0.0f, 0.0f, 0.0f);
	//transform->mRotation = glm::quat(0.0f, 0.0f, 0.0f, 1.0f);
	//transform->mScale = glm::vec3(1.0f, 1.0f, 1.0f);

	//Camera* camera = cameraEntity->addComponent<Camera>(&previewWorld);
	//camera->mPosition = glm::vec3(3.0f, 0.0f, 0.0f);
	//camera->mFront = glm::vec3(-3.0f, 0.0f, 0.0f);
	//camera->mUp = glm::vec3(0.0f, 0.0f, 1.0f);

	//// add render system to material view world
	//RenderSystem* renderSystem = previewWorld.addSystem<RenderSystem>(0);
	//renderSystem->mRenderToScreen = false;

	//// add required clean-up system to material view world
	//CleanUpSystem* cleanUpSystem = previewWorld.addSystem<CleanUpSystem>(1);
}




























//void MaterialDrawer::initViewWorld()
//{
//	// create shader in material view world
//	previewShader = previewWorld.createAsset<Shader>();
//
//	// create material in material view world
//	previewMaterial = previewWorld.createAsset<Material>();
//	previewMaterial->setShaderId(previewShader->getId());
//
//	// create sphere mesh in material view world
//	Mesh* sphereMesh = previewWorld.createAsset<Mesh>();
//	sphereMesh->load(PhysicsEngine::InternalMeshes::sphereVertices,
//					 PhysicsEngine::InternalMeshes::sphereNormals,
//					 PhysicsEngine::InternalMeshes::sphereTexCoords,
//					 PhysicsEngine::InternalMeshes::sphereSubMeshStartIndicies);
//
//	// create sphere entity in material view world
//	Entity* sphereEntity = previewWorld.createEntity();
//
//	Transform* transform = sphereEntity->addComponent<Transform>(&previewWorld);
//	transform->mPosition = glm::vec3(0.0f, 0.0f, 0.0f);
//	transform->mRotation = glm::quat(0.0f, 0.0f, 0.0f, 1.0f);
//	transform->mScale = glm::vec3(1.0f, 1.0f, 1.0f);
//
//	MeshRenderer* meshRenderer = sphereEntity->addComponent<MeshRenderer>(&previewWorld);
//	meshRenderer->mMeshId = sphereMesh->getId();
//	meshRenderer->mMaterialIds[0] = previewMaterial->getId();
//	meshRenderer->mMaterialCount = 1;
//	meshRenderer->mIsStatic = false;
//
//	// create light entity in material view world
//	Entity* lightEntity = previewWorld.createEntity();
//
//	transform = lightEntity->addComponent<Transform>(&previewWorld);
//	transform->mPosition = glm::vec3(0.0f, 0.0f, 0.0f);
//	transform->mRotation = glm::quat(0.0f, 0.0f, 0.0f, 1.0f);
//	transform->mScale = glm::vec3(1.0f, 1.0f, 1.0f);
//
//	Light* light = lightEntity->addComponent<Light>(&previewWorld);
//
//	// create camera entity in material view world
//	Entity* cameraEntity = previewWorld.createEntity();
//
//	transform = cameraEntity->addComponent<Transform>(&previewWorld);
//	transform->mPosition = glm::vec3(0.0f, 0.0f, 0.0f);
//	transform->mRotation = glm::quat(0.0f, 0.0f, 0.0f, 1.0f);
//	transform->mScale = glm::vec3(1.0f, 1.0f, 1.0f);
//
//	Camera* camera = cameraEntity->addComponent<Camera>(&previewWorld);
//	camera->mPosition = glm::vec3(3.0f, 0.0f, 0.0f);
//	camera->mFront = glm::vec3(-3.0f, 0.0f, 0.0f);
//	camera->mUp = glm::vec3(0.0f, 0.0f, 1.0f);
//
//	// add render system to material view world
//	RenderSystem* renderSystem = previewWorld.addSystem<RenderSystem>(0);
//	renderSystem->mRenderToScreen = false;
//
//	// add required clean-up system to material view world
//	CleanUpSystem* cleanUpSystem = previewWorld.addSystem<CleanUpSystem>(1);
//
//	for (int i = 0; i < previewWorld.getNumberOfSystems(); i++) {
//		System* system = previewWorld.getSystemByIndex(i);
//
//		system->init(&previewWorld);
//	}
//}
//
//void MaterialDrawer::updateViewWorld()
//{
//	for (int i = 0; i < previewWorld.getNumberOfSystems(); i++) {
//		System* system = previewWorld.getSystemByIndex(i);
//
//		system->update({});
//	}
//
//	Log::info("\n");
//	Log::info(("material view world system count: " + std::to_string(previewWorld.getNumberOfSystems()) + "\n").c_str());
//	Log::info(("material view world entity count: " + std::to_string(previewWorld.getNumberOfEntities()) + "\n").c_str());
//	Log::info(("material view world material count: " + std::to_string(previewWorld.getNumberOfAssets<Material>()) + "\n").c_str());
//	Log::info(("material view world mesh count: " + std::to_string(previewWorld.getNumberOfAssets<Mesh>()) + "\n").c_str());
//	Log::info(("material view world shader count: " + std::to_string(previewWorld.getNumberOfAssets<Shader>()) + "\n").c_str());
//	Log::info(("material view world texture count: " + std::to_string(previewWorld.getNumberOfAssets<Texture2D>()) + "\n").c_str());
//
//	RenderSystem* renderSystem = previewWorld.getSystem<RenderSystem>();
//
//	GraphicsTargets targets = renderSystem->getGraphicsTargets();
//
//	ImGuiWindowFlags window_flags = ImGuiWindowFlags_None;// ImGuiWindowFlags_HorizontalScrollbar | (disable_mouse_wheel ? ImGuiWindowFlags_NoScrollWithMouse : 0);
//	ImGui::BeginChild("MaterialPreviewWindow", ImVec2(ImGui::GetWindowContentRegionWidth(), ImGui::GetWindowContentRegionWidth()), true, window_flags);
//	ImGui::Image((void*)(intptr_t)targets.mColor, ImVec2(ImGui::GetWindowContentRegionWidth(), ImGui::GetWindowContentRegionWidth()), ImVec2(1, 1), ImVec2(0, 0));
//	ImGui::EndChild();
//}

//void MaterialDrawer::initViewWorld(World* world, Material* material)
//{
//	Shader* shader = world->getAsset<Shader>(material->getShaderId());
//
//	// create shader in material view world
//	Shader* viewShader = viewWorld.createAssetWithId<Shader>(shader->getId());
//	viewShader->load(shader->getVertexShader(), shader->getFragmentShader(), shader->getGeometryShader());
//	viewShader->compile();
//
//	// create material in material view world
//	viewMaterial = viewWorld.createAsset<Material>();
//	viewMaterial->setShaderId(viewShader->getId());
//	viewMaterial->onShaderChanged(&viewWorld);
//
//	// create sphere mesh in material view world
//	Mesh* sphereMesh = viewWorld.createAsset<Mesh>();
//	sphereMesh->load(PhysicsEngine::InternalMeshes::sphereVertices,
//					 PhysicsEngine::InternalMeshes::sphereNormals,
//					 PhysicsEngine::InternalMeshes::sphereTexCoords,
//					 PhysicsEngine::InternalMeshes::sphereSubMeshStartIndicies);
//
//	// create sphere entity in material view world
//	Entity* sphereEntity = viewWorld.createEntity();
//
//	Transform* transform = sphereEntity->addComponent<Transform>(&viewWorld);
//	transform->mPosition = glm::vec3(0.0f, 0.0f, 0.0f);
//	transform->mRotation = glm::quat(0.0f, 0.0f, 0.0f, 1.0f);
//	transform->mScale = glm::vec3(1.0f, 1.0f, 1.0f);
//
//	MeshRenderer* meshRenderer = sphereEntity->addComponent<MeshRenderer>(&viewWorld);
//	meshRenderer->mMeshId = sphereMesh->getId();
//	meshRenderer->mMaterialIds[0] = viewMaterial->getId();
//	meshRenderer->mMaterialCount = 1;
//	meshRenderer->mIsStatic = false;
//
//	// create light entity in material view world
//	Entity* lightEntity = viewWorld.createEntity();
//
//	transform = lightEntity->addComponent<Transform>(&viewWorld);
//	transform->mPosition = glm::vec3(0.0f, 0.0f, 0.0f);
//	transform->mRotation = glm::quat(0.0f, 0.0f, 0.0f, 1.0f);
//	transform->mScale = glm::vec3(1.0f, 1.0f, 1.0f);
//
//	Light* light = lightEntity->addComponent<Light>(&viewWorld);
//
//	// create camera entity in material view world
//	Entity* cameraEntity = viewWorld.createEntity();
//
//	transform = cameraEntity->addComponent<Transform>(&viewWorld);
//	transform->mPosition = glm::vec3(0.0f, 0.0f, 0.0f);
//	transform->mRotation = glm::quat(0.0f, 0.0f, 0.0f, 1.0f);
//	transform->mScale = glm::vec3(1.0f, 1.0f, 1.0f);
//
//	Camera* camera = cameraEntity->addComponent<Camera>(&viewWorld);
//	camera->mPosition = glm::vec3(3.0f, 0.0f, 0.0f);
//	camera->mFront = glm::vec3(-3.0f, 0.0f, 0.0f);
//	camera->mUp = glm::vec3(0.0f, 0.0f, 1.0f);
//
//	// add render system to material view world
//	RenderSystem* renderSystem = viewWorld.addSystem<RenderSystem>(0);
//	renderSystem->mRenderToScreen = false;
//
//	// add required clean-up system to material view world
//	CleanUpSystem* cleanUpSystem = viewWorld.addSystem<CleanUpSystem>(1);
//
//	for (int i = 0; i < viewWorld.getNumberOfSystems(); i++) {
//		System* system = viewWorld.getSystemByIndex(i);
//
//		system->init(&viewWorld);
//	}
//}

//void MaterialDrawer::updateViewWorld(World* world, Material* material)
//{
//	// Copy uniforms from source material to sphere material
//	std::vector<ShaderUniform> uniforms = material->getUniforms();
//	for (size_t i = 0; i < uniforms.size(); i++) {
//
//		// Note: matrices not supported
//		switch (uniforms[i].mType)
//		{
//		case GL_INT:
//			viewMaterial->setInt(uniforms[i].mName, material->getInt(uniforms[i].mName));
//			break;
//		case GL_FLOAT:
//			viewMaterial->setFloat(uniforms[i].mName, material->getFloat(uniforms[i].mName));
//			break;
//		case GL_FLOAT_VEC2:
//			viewMaterial->setVec2(uniforms[i].mName, material->getVec2(uniforms[i].mName));
//			break;
//		case GL_FLOAT_VEC3:
//			viewMaterial->setVec3(uniforms[i].mName, material->getVec3(uniforms[i].mName));
//			break;
//		case GL_FLOAT_VEC4:
//			viewMaterial->setVec4(uniforms[i].mName, material->getVec4(uniforms[i].mName));
//			break;
//		case GL_SAMPLER_2D:
//			Guid textureId = material->getTexture(uniforms[i].mName);
//			if (textureId != Guid::INVALID) {
//				Texture2D* texture = world->getAsset<Texture2D>(textureId);
//				if (texture != NULL) {
//					Texture2D* viewTexture = viewWorld.getAsset<Texture2D>(textureId);
//					if (viewTexture == NULL) {
//						viewTexture = viewWorld.createAssetWithId<Texture2D>(textureId);
//					}
//
//					viewTexture->setRawTextureData(texture->getRawTextureData(), 
//												   texture->getWidth(), 
//												   texture->getHeight(), 
//												   texture->getFormat());
//
//					viewMaterial->setTexture(uniforms[i].mName, viewTexture->getId());
//				}
//			}
//			break;
//		}
//	}
//
//	for (int i = 0; i < viewWorld.getNumberOfSystems(); i++) {
//		System* system = viewWorld.getSystemByIndex(i);
//
//		system->update({});
//	}
//
//	/*Log::info("\n");
//	Log::info(("material view world system count: " + std::to_string(viewWorld.getNumberOfSystems()) + "\n").c_str());
//	Log::info(("material view world entity count: " + std::to_string(viewWorld.getNumberOfEntities()) + "\n").c_str());
//	Log::info(("material view world material count: " + std::to_string(viewWorld.getNumberOfAssets<Material>()) + "\n").c_str());
//	Log::info(("material view world mesh count: " + std::to_string(viewWorld.getNumberOfAssets<Mesh>()) + "\n").c_str());
//	Log::info(("material view world shader count: " + std::to_string(viewWorld.getNumberOfAssets<Shader>()) + "\n").c_str());
//	Log::info(("material view world texture count: " + std::to_string(viewWorld.getNumberOfAssets<Texture2D>()) + "\n").c_str());*/
//
//	RenderSystem* renderSystem = viewWorld.getSystem<RenderSystem>();
//
//	GraphicsTargets targets = renderSystem->getGraphicsTargets();
//
//	ImGuiWindowFlags window_flags = ImGuiWindowFlags_None;// ImGuiWindowFlags_HorizontalScrollbar | (disable_mouse_wheel ? ImGuiWindowFlags_NoScrollWithMouse : 0);
//	ImGui::BeginChild("MaterialPreviewWindow", ImVec2(ImGui::GetWindowContentRegionWidth(), ImGui::GetWindowContentRegionWidth()), true, window_flags);
//	ImGui::Image((void*)(intptr_t)targets.mColor, ImVec2(ImGui::GetWindowContentRegionWidth(), ImGui::GetWindowContentRegionWidth()), ImVec2(1, 1), ImVec2(0, 0));
//	ImGui::EndChild();
//}