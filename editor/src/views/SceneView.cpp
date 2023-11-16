#include "../../include/views/SceneView.h"
#include "../../include/ProjectDatabase.h"

#include "core/Intersect.h"
#include "core/Log.h"
#include "core/Rect.h"
#include "core/Application.h"
#include "graphics/RenderContext.h"

#include "../../include/imgui/imgui_extensions.h"

#include <chrono>

using namespace PhysicsEditor;

SceneView::SceneView() : mOpen(true), mFocused(false), mHovered(false), mHoveredLastFrame(false)
{
	mActiveDebugTarget = DebugTargets::Color;
	mOperation = ImGuizmo::OPERATION::TRANSLATE;
	mCoordinateMode = ImGuizmo::MODE::LOCAL;

	mSceneContentMin = ImVec2(0, 0);
	mSceneContentMax = ImVec2(0, 0);
	mSceneContentSize = ImVec2(0, 0);
	mIsSceneContentHovered = false;
}

SceneView::~SceneView()
{
}

void SceneView::init(Clipboard& clipboard)
{
	initWorld(clipboard.getWorld());
}

void SceneView::update(Clipboard& clipboard, bool isOpenedThisFrame)
{
	mHoveredLastFrame = mHovered;
	mFocused = false;
	mHovered = false;

	if (isOpenedThisFrame)
	{
		mOpen = true;
	}

	if (!mOpen)
	{
		return;
	}

	if (ImGui::Begin("SceneView", &mOpen))
	{
		if (ImGui::GetIO().MouseClicked[1] && ImGui::IsWindowHovered())
		{
			ImGui::SetWindowFocus("SceneView");
		}
	}

	mWindowPos = ImGui::GetWindowPos();
	mWindowWidth = ImGui::GetWindowWidth();
	mWindowHeight = ImGui::GetWindowHeight();
	mContentMin = ImGui::GetWindowContentRegionMin();
	mContentMax = ImGui::GetWindowContentRegionMax();

	mContentMin.x += mWindowPos.x;
	mContentMin.y += mWindowPos.y;
	mContentMax.x += mWindowPos.x;
	mContentMax.y += mWindowPos.y;

	mFocused = ImGui::IsWindowFocused();
	mHovered = ImGui::IsWindowHovered(ImGuiHoveredFlags_AllowWhenBlockedByActiveItem);

	/*clipboard.mOpen[static_cast<int>(View::SceneView)] = isOpen();
	clipboard.mHovered[static_cast<int>(View::SceneView)] = isHovered();
	clipboard.mFocused[static_cast<int>(View::SceneView)] = isFocused();
	clipboard.mOpenedThisFrame[static_cast<int>(View::SceneView)] = openedThisFrame();
	clipboard.mHoveredThisFrame[static_cast<int>(View::SceneView)] = hoveredThisFrame();
	clipboard.mFocusedThisFrame[static_cast<int>(View::SceneView)] = focusedThisFrame();
	clipboard.mClosedThisFrame[static_cast<int>(View::SceneView)] = closedThisFrame();
	clipboard.mUnfocusedThisFrame[static_cast<int>(View::SceneView)] = unfocusedThisFrame();
	clipboard.mUnhoveredThisFrame[static_cast<int>(View::SceneView)] = unhoveredThisFrame();*/

	mSceneContentMin = getContentMin();
	mSceneContentMax = getContentMax();

	// account for the fact that Image will draw below buttons
	mSceneContentMin.y += 23;

	mSceneContentSize.x = mSceneContentMax.x - mSceneContentMin.x;
	mSceneContentSize.y = mSceneContentMax.y - mSceneContentMin.y;

	ImGuiIO& io = ImGui::GetIO();
	PhysicsEngine::Rect sceneContentRect(mSceneContentMin.x, mSceneContentMin.y, mSceneContentSize.x, mSceneContentSize.y);
	mIsSceneContentHovered = sceneContentRect.contains(io.MousePos.x, io.MousePos.y);

	if (clipboard.mProjectOpened)
	{
		updateWorld(clipboard.getWorld());

		// Do stuff....
		drawSceneHeader(clipboard);

		if (clipboard.mSceneOpened)
		{
			// Do more stuff...
			drawSceneContent(clipboard);
		}
	}

	ImGui::End();
}

void SceneView::drawSceneHeader(Clipboard& clipboard)
{
	static bool gizmosChecked = false;
	static bool overlayChecked = false;
	static bool vsyncChecked = true;
	static bool cameraSettingsClicked = false;
	static bool translationModeActive = true;
	static bool rotationModeActive = false;
	static bool scaleModeActive = false;

	const char* targetNames[] = { "Color", "Color Picking", "Depth", "Linear Depth", "Normals",
									"Shadow Cascades", "Position", "Albedo/Specular", "SSAO",  "SSAO Noise", "OcclusionMap"};

	clipboard.mCameraSystem->setViewport(0, 0, (int)mSceneContentSize.x, (int)mSceneContentSize.y);

	// select draw texture dropdown
	ImGui::PushItemWidth(0.25f * ImGui::GetWindowSize().x);
	if (ImGui::BeginCombo("##DrawTexture", targetNames[static_cast<int>(mActiveDebugTarget)]))
	{
		for (int n = 0; n < static_cast<int>(DebugTargets::Count); n++)
		{
			bool is_selected = (mActiveDebugTarget == static_cast<DebugTargets>(n));
			if (ImGui::Selectable(targetNames[n], is_selected))
			{
				mActiveDebugTarget = static_cast<DebugTargets>(n);

				clipboard.mCameraSystem->getCamera()->mColorTarget = PhysicsEngine::ColorTarget::Color;
				if (mActiveDebugTarget == DebugTargets::Normals)
				{
					clipboard.mCameraSystem->getCamera()->mColorTarget = PhysicsEngine::ColorTarget::Normal;
				}
				else if (mActiveDebugTarget == DebugTargets::Position)
				{
					clipboard.mCameraSystem->getCamera()->mColorTarget = PhysicsEngine::ColorTarget::Position;
				}
				else if (mActiveDebugTarget == DebugTargets::LinearDepth)
				{
					clipboard.mCameraSystem->getCamera()->mColorTarget = PhysicsEngine::ColorTarget::LinearDepth;
				}
				else if (mActiveDebugTarget == DebugTargets::ShadowCascades)
				{
					clipboard.mCameraSystem->getCamera()->mColorTarget = PhysicsEngine::ColorTarget::ShadowCascades;
				}

				if (is_selected)
				{
					ImGui::SetItemDefaultFocus();
				}
			}
		}
		ImGui::EndCombo();
	}
	ImGui::PopItemWidth();
	ImGui::SameLine();

	// whether to render gizmos or not
	if (ImGui::Checkbox("Gizmos", &gizmosChecked))
	{
		clipboard.mCameraSystem->setGizmos(gizmosChecked ? PhysicsEngine::CameraGizmos::Gizmos_On : PhysicsEngine::CameraGizmos::Gizmos_Off);
	}
	ImGui::SameLine();

	// editor rendering performance overlay
	if (ImGui::Checkbox("Perf", &overlayChecked))
	{
	}
	ImGui::SameLine();

	if (ImGui::Checkbox("VSYNC", &vsyncChecked))
	{
		if (vsyncChecked)
		{
			PhysicsEngine::Application::get().getWindow().turnVsyncOn();
		}
		else
		{
			PhysicsEngine::Application::get().getWindow().turnVsyncOff();
		}
	}
	ImGui::SameLine();

	// select transform gizmo movement mode
	if (ImGui::StampButton("T", translationModeActive))
	{
		translationModeActive = true;
		rotationModeActive = false;
		scaleModeActive = false;
		mOperation = ImGuizmo::OPERATION::TRANSLATE;
	}
	ImGui::SameLine();

	if (ImGui::StampButton("R", rotationModeActive))
	{
		translationModeActive = false;
		rotationModeActive = true;
		scaleModeActive = false;
		mOperation = ImGuizmo::OPERATION::ROTATE;
	}
	ImGui::SameLine();

	if (ImGui::StampButton("S", scaleModeActive))
	{
		translationModeActive = false;
		rotationModeActive = false;
		scaleModeActive = true;
		mOperation = ImGuizmo::OPERATION::SCALE;
	}
	ImGui::SameLine();

	std::vector<std::string> worldLocalNames = { "Local", "World" };
	//const char* worldLocalNames[] = { "Local", "World" };
	ImGui::PushItemWidth(0.1f * ImGui::GetWindowSize().x);
	static int coordinateMode = static_cast<int>(mCoordinateMode);
	if (ImGui::Combo("##Coordinate Mode", &coordinateMode, worldLocalNames))
	{
		mCoordinateMode = static_cast<ImGuizmo::MODE>(coordinateMode);
	}
	ImGui::PopItemWidth();
	ImGui::SameLine();

	// editor camera settings
	if (ImGui::Button("Camera Settings"))
	{
		cameraSettingsClicked = true;
	}

	if (cameraSettingsClicked)
	{
		drawCameraSettingsPopup(clipboard.mCameraSystem, &cameraSettingsClicked);
	}

	// performance overlay
	if (overlayChecked)
	{
		drawPerformanceOverlay(clipboard, clipboard.mCameraSystem);
	}
}

void SceneView::drawSceneContent(Clipboard& clipboard)
{
	ImGuiIO& io = ImGui::GetIO();
	float mousePosX = std::min(std::max(io.MousePos.x - mSceneContentMin.x, 0.0f), mSceneContentSize.x);
	float mousePosY = mSceneContentSize.y - std::min(std::max(io.MousePos.y - mSceneContentMin.y, 0.0f), mSceneContentSize.y);

	clipboard.mGizmoSystem->mEnabled = true;

	// Update selected entity
	if (isSceneContentHovered() && io.MouseClicked[0] && !ImGuizmo::IsOver())
	{
		float nx = mousePosX / mSceneContentSize.x;
		float ny = mousePosY / mSceneContentSize.y;

		PhysicsEngine::Id transformId = clipboard.mCameraSystem->getTransformUnderMouse(nx, ny);

		PhysicsEngine::Transform* transform = clipboard.getWorld()->getActiveScene()->getComponentById<PhysicsEngine::Transform>(transformId);

		if (transform != nullptr)
		{
			clipboard.setSelectedItem(InteractionType::Entity, transform->getEntityGuid());
		}
		else
		{
			clipboard.setSelectedItem(InteractionType::None, PhysicsEngine::Guid::INVALID);
		}
	}

	clipboard.mGizmoSystem->clearDrawList();

	// draw camera gizmos
	for (size_t i = 0; i < clipboard.mWorld.getActiveScene()->getNumberOfComponents<PhysicsEngine::Camera>(); i++)
	{
		PhysicsEngine::Camera* camera = clipboard.mWorld.getActiveScene()->getComponentByIndex<PhysicsEngine::Camera>(i);

		if (camera->mHide == PhysicsEngine::HideFlag::None && camera->mEnabled)
		{
			PhysicsEngine::Entity* entity = camera->getEntity();
			PhysicsEngine::Transform* transform = clipboard.mWorld.getActiveScene()->getComponent<PhysicsEngine::Transform>(entity->getGuid());

			glm::vec3 position = transform->getPosition();
			glm::vec3 front = transform->getForward();
			glm::vec3 up = transform->getUp();
			glm::vec3 right = transform->getRight();

			std::array<PhysicsEngine::Color, 5> cascadeColors = { PhysicsEngine::Color::red,
																  PhysicsEngine::Color::green,
																  PhysicsEngine::Color::blue,
																  PhysicsEngine::Color::cyan,
																  PhysicsEngine::Color::magenta };

			std::array<PhysicsEngine::Frustum, 5> cascadeFrustums = camera->calcCascadeFrustums(camera->calcViewSpaceCascadeEnds());
			for (size_t j = 0; j < cascadeFrustums.size(); j++)
			{
				cascadeColors[j].mA = 0.3f;
				cascadeFrustums[j].computePlanes(position, front, up, right);
				clipboard.mGizmoSystem->addToDrawList(cascadeFrustums[j], cascadeColors[j], false);
			}
		}
	}

	// drag n drop meshes into scene
	const ImGuiPayload* peek = ImGui::GetDragDropPayload();
	if (peek != nullptr && peek->IsDataType("MESH_PATH"))
	{
		if (hoveredThisFrame())
		{
			const char* data = static_cast<const char*>(peek->Data);
			std::filesystem::path incomingPath = std::string(data);

			PhysicsEngine::Entity* entity = clipboard.getWorld()->getActiveScene()->createNonPrimitive(ProjectDatabase::getGuid(incomingPath));
			PhysicsEngine::Transform* transform = entity->getComponent<PhysicsEngine::Transform>();

			clipboard.mSceneViewTempEntity = entity;
			clipboard.mSceneViewTempTransform = transform;
		}

		if (unhoveredThisFrame())
		{
			if (clipboard.mSceneViewTempEntity != nullptr)
			{
				clipboard.getWorld()->getActiveScene()->immediateDestroyEntity(clipboard.mSceneViewTempEntity->getGuid());
				clipboard.mSceneViewTempEntity = nullptr;
				clipboard.mSceneViewTempTransform = nullptr;
			}
		}

		if (isHovered())
		{
			if (clipboard.mSceneViewTempEntity != nullptr)
			{
				float ndc_x = 2 * (mousePosX - 0.5f * mSceneContentSize.x) / mSceneContentSize.x;
				float ndc_y = 2 * (mousePosY - 0.5f * mSceneContentSize.y) / mSceneContentSize.y;

				PhysicsEngine::Ray cameraRay = clipboard.mCameraSystem->normalizedDeviceSpaceToRay(ndc_x, ndc_y);

				PhysicsEngine::Plane xz;
				xz.mX0 = glm::vec3(0, 0, 0);
				xz.mNormal = glm::vec3(0, 1, 0);

				float dist = -1.0f;
				bool intersects = PhysicsEngine::Intersect::intersect(cameraRay, xz, dist);

				clipboard.mSceneViewTempTransform->setPosition((intersects && dist >= 0.0f) ? cameraRay.getPoint(dist) : cameraRay.getPoint(5.0f));
			}
		}
	}

	// Finally draw scene
	PhysicsEngine::RenderTextureHandle* tex = clipboard.mCameraSystem->getNativeGraphicsColorTex();

	switch (mActiveDebugTarget)
	{
	case DebugTargets::Depth:
		tex = clipboard.mCameraSystem->getNativeGraphicsDepthTex();
		break;
	case DebugTargets::ColorPicking:
		tex = clipboard.mCameraSystem->getNativeGraphicsColorPickingTex();
		break;
	case DebugTargets::AlbedoSpecular:
		tex = clipboard.mCameraSystem->getNativeGraphicsAlbedoSpecTex();
		break;
	case DebugTargets::SSAO:
		tex = clipboard.mCameraSystem->getNativeGraphicsSSAOColorTex();
		break;
	case DebugTargets::SSAONoise:
		tex = clipboard.mCameraSystem->getNativeGraphicsSSAONoiseTex();
		break;
	case DebugTargets::OcclusionMap:
		tex = clipboard.mCameraSystem->getNativeGraphicsOcclusionMapTex();
		break;
	}

	if (tex != nullptr)
	{
		if (PhysicsEngine::RenderContext::getRenderAPI() == PhysicsEngine::RenderAPI::OpenGL)
		{
			ImVec2 uv0 = ImVec2(0, std::min(1.0f, mSceneContentSize.y / tex->getHeight()));
			ImVec2 uv1 = ImVec2(std::min(1.0f, mSceneContentSize.x / tex->getWidth()), 0);

			// opengl
			ImGui::Image((void*)(intptr_t)(*reinterpret_cast<unsigned int*>(tex->getIMGUITexture())), mSceneContentSize, uv0, uv1);
		}
		else
		{
			ImVec2 uv0 = ImVec2(0, 0);
			ImVec2 uv1 = ImVec2(std::min(1.0f, mSceneContentSize.x / tex->getWidth()), std::min(1.0f, mSceneContentSize.y / tex->getHeight()));

			// directx
			ImGui::Image(tex->getIMGUITexture(), mSceneContentSize, uv0, uv1);
		}
	}

	// draw transform gizmo if entity is selected
	if (clipboard.getSelectedType() == InteractionType::Entity)
	{
		PhysicsEngine::Transform* transform = clipboard.getWorld()->getActiveScene()->getComponent<PhysicsEngine::Transform>(clipboard.getSelectedId());

		if (transform != nullptr)
		{
			ImGuizmo::SetOrthographic(false);
			ImGuizmo::SetDrawlist();
			float windowWidth = ImGui::GetWindowWidth();
			float windowHeight = ImGui::GetWindowHeight();
			ImGuizmo::SetRect(ImGui::GetWindowPos().x, ImGui::GetWindowPos().y, windowWidth, windowHeight);

			glm::mat4 view = clipboard.mCameraSystem->getViewMatrix();
			glm::mat4 projection = clipboard.mCameraSystem->getProjMatrix();
			glm::mat4 model = transform->getModelMatrix();

			ImGuizmo::AllowAxisFlip(false);

			ImGuizmo::Manipulate(glm::value_ptr(view), glm::value_ptr(projection), mOperation,
				mCoordinateMode, glm::value_ptr(model), NULL, NULL);

			if (ImGuizmo::IsUsing())
			{
				glm::vec3 scale;
				glm::quat rotation;
				glm::vec3 translation;

				PhysicsEngine::Transform::decompose(model, translation, rotation, scale);

				transform->setPosition(translation);
				transform->setScale(scale);
				transform->setRotation(rotation);
			}

			PhysicsEngine::Camera* camera = clipboard.getWorld()->getActiveScene()->getComponent<PhysicsEngine::Camera>(clipboard.getSelectedId());
			if (camera != nullptr && camera->mEnabled)
			{
				camera->computeViewMatrix(transform->getPosition(), transform->getForward(), transform->getUp(), transform->getRight());

				ImVec2 min = mSceneContentMin;
				ImVec2 max = mSceneContentMax;

				min.x += 0.6f * getWindowWidth();
				min.y += 0.6f * getWindowHeight();

				if (camera->mRenderTextureId.isValid())
				{
					PhysicsEngine::RenderTexture* renderTexture = clipboard.getWorld()->getAssetByGuid<PhysicsEngine::RenderTexture>(camera->mRenderTextureId);
					if (renderTexture->getNativeGraphicsColorTex()->getIMGUITexture() != nullptr)
					{
						ImGui::GetWindowDrawList()->AddImage((void*)(intptr_t)(*reinterpret_cast<unsigned int*>(renderTexture->getNativeGraphicsColorTex()->getIMGUITexture())), min, max, ImVec2(0, 1), ImVec2(1, 0));
					}
				}
				else
				{
					if (camera->getNativeGraphicsColorTex()->getIMGUITexture() != nullptr)
					{
						ImGui::GetWindowDrawList()->AddImage((void*)(intptr_t)(*reinterpret_cast<unsigned int*>(camera->getNativeGraphicsColorTex()->getIMGUITexture())), min, max, ImVec2(0, 1), ImVec2(1, 0));
					}
				}
			}
		}
	}
}

ImVec2 SceneView::getSceneContentMin() const
{
	return mSceneContentMin;
}

ImVec2 SceneView::getSceneContentMax() const
{
	return mSceneContentMax;
}

bool SceneView::isSceneContentHovered() const
{
	return mIsSceneContentHovered;
}

void SceneView::initWorld(PhysicsEngine::World* world)
{
	PhysicsEngine::FreeLookCameraSystem* cameraSystem = world->getSystem<PhysicsEngine::FreeLookCameraSystem>();
	PhysicsEngine::TerrainSystem* terrainSystem = world->getSystem<PhysicsEngine::TerrainSystem>();
	PhysicsEngine::RenderSystem* renderSystem = world->getSystem<PhysicsEngine::RenderSystem>();
	PhysicsEngine::GizmoSystem* gizmoSystem = world->getSystem<PhysicsEngine::GizmoSystem>();
	PhysicsEngine::CleanUpSystem* cleanUpSystem = world->getSystem<PhysicsEngine::CleanUpSystem>();

	assert(cameraSystem != nullptr);
	assert(terrainSystem != nullptr);
	assert(renderSystem != nullptr);
	assert(gizmoSystem != nullptr);
	assert(cleanUpSystem != nullptr);

	cameraSystem->init(world);
	terrainSystem->init(world);
	renderSystem->init(world);
	gizmoSystem->init(world);
	cleanUpSystem->init(world);
}

void SceneView::updateWorld(PhysicsEngine::World* world)
{
	ImGuiIO& io = ImGui::GetIO();

	PhysicsEngine::Input& input = PhysicsEngine::getInput();

	// Mouse
	if (isFocused())
	{
		// clamp mouse position to be within the scene view content region
		input.mMousePosX = std::min(std::max((int)io.MousePos.x - (int)mSceneContentMin.x, 0), (int)mSceneContentSize.x);
		input.mMousePosY =
			(int)mSceneContentSize.y -
			std::min(std::max((int)io.MousePos.y - (int)mSceneContentMin.y, 0), (int)mSceneContentSize.y);
	}
	
	// call update on all systems in world
	PhysicsEngine::FreeLookCameraSystem* cameraSystem = world->getSystem<PhysicsEngine::FreeLookCameraSystem>();
	PhysicsEngine::TerrainSystem* terrainSystem = world->getSystem<PhysicsEngine::TerrainSystem>();
	PhysicsEngine::RenderSystem* renderSystem = world->getSystem<PhysicsEngine::RenderSystem>();
	PhysicsEngine::GizmoSystem* gizmoSystem = world->getSystem<PhysicsEngine::GizmoSystem>();
	PhysicsEngine::CleanUpSystem* cleanUpSystem = world->getSystem<PhysicsEngine::CleanUpSystem>();

	assert(cameraSystem != nullptr);
	assert(terrainSystem != nullptr);
	assert(renderSystem != nullptr);
	assert(gizmoSystem != nullptr);
	assert(cleanUpSystem != nullptr);

	if (cameraSystem->mEnabled) {
		cameraSystem->update();
	}
	if (terrainSystem->mEnabled) {
		terrainSystem->update();
	}
	if (renderSystem->mEnabled) {
		renderSystem->update();
	}
	if (gizmoSystem->mEnabled) {
		gizmoSystem->update();
	}
	if (cleanUpSystem->mEnabled) {
		cleanUpSystem->update();
	}
}

void SceneView::drawPerformanceOverlay(Clipboard& clipboard, PhysicsEngine::FreeLookCameraSystem* cameraSystem)
{
	static bool overlayOpened = false;
	static ImGuiWindowFlags overlayFlags = ImGuiWindowFlags_Tooltip | ImGuiWindowFlags_NoTitleBar |
		ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoSavedSettings |
		ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoDocking |
		ImGuiWindowFlags_NoNav | ImGuiWindowFlags_NoMove;

	ImVec2 overlayPos = ImVec2(mSceneContentMax.x, mSceneContentMin.y);

	ImGui::SetNextWindowPos(overlayPos, ImGuiCond_Always, ImVec2(1.0f, 0.0f));
	ImGui::SetNextWindowBgAlpha(0.35f); // Transparent background
	if (ImGui::Begin("Editor Performance Overlay", &overlayOpened, overlayFlags))
	{
		ImGuiIO& io = ImGui::GetIO();

		ImGui::Text("Project name: %s\n", clipboard.getProjectName().c_str());
		ImGui::Text("Project path: %s\n", clipboard.getProjectPath().string().c_str());
		ImGui::Text("Scene name: %s\n", clipboard.getSceneName().c_str());
		ImGui::Text("Scene path: %s\n", clipboard.getScenePath().string().c_str());

		ImGui::Text("Tris: %d\n", cameraSystem->getQuery().mTris);
		ImGui::Text("Verts: %d\n", cameraSystem->getQuery().mVerts);
		ImGui::Text("Draw calls: %d\n", cameraSystem->getQuery().mNumDrawCalls);
		ImGui::Text("Instance Draw calls: %d\n", cameraSystem->getQuery().mNumInstancedDrawCalls);
		ImGui::Text("Elapsed time: %f", cameraSystem->getQuery().mTotalElapsedTime);
		ImGui::Text("Frame count: %d", clipboard.mTime.mFrameCount);
		ImGui::Text("Delta time: %f", clipboard.mTime.mDeltaTime);
		ImGui::Text("getSmoothedDeltaTime: %f", PhysicsEngine::getSmoothedDeltaTime(clipboard.mTime));
		ImGui::Text("getFPS: %f", PhysicsEngine::getFPS(clipboard.mTime));
		ImGui::Text("getSmoothedFPS: %f", PhysicsEngine::getSmoothedFPS(clipboard.mTime));
		ImGui::Text("ImGui::GetIO().Framerate: %f", ImGui::GetIO().Framerate);
		ImGui::Text("Window position: %f %f\n", getWindowPos().x, getWindowPos().y);
		ImGui::Text("Scene content min: %f %f\n", mSceneContentMin.x, mSceneContentMin.y);
		ImGui::Text("Scene content max: %f %f\n", mSceneContentMax.x, mSceneContentMax.y);
		ImGui::Text("Is Scene content hovered: %d\n", mIsSceneContentHovered);
		ImGui::Text("Mouse Position: %d %d\n", cameraSystem->getMousePosX(), cameraSystem->getMousePosY());
		ImGui::Text("Mouse Position: %f %f\n", io.MousePos.x, io.MousePos.y);
		ImGui::Text("Normalized Mouse Position: %f %f\n",
			cameraSystem->getMousePosX() / (float)(mSceneContentMax.x - mSceneContentMin.x),
			cameraSystem->getMousePosY() / (float)(mSceneContentMax.y - mSceneContentMin.y));

		ImGui::Text("Selected interaction type %d\n", clipboard.getSelectedType());
		ImGui::Text("Selected id %s\n", clipboard.getSelectedId().toString().c_str());
		ImGui::Text("Selected path %s\n", clipboard.getSelectedPath().c_str());

		float width = (float)(mSceneContentMax.x - mSceneContentMin.x);
		float height = (float)(mSceneContentMax.y - mSceneContentMin.y);
		ImGui::Text("NDC: %f %f\n", 2 * (cameraSystem->getMousePosX() - 0.5f * width) / width,
			2 * (cameraSystem->getMousePosY() - 0.5f * height) / height);

		ImGui::Text("Camera Position: %f %f %f\n", cameraSystem->getCameraPosition().x, cameraSystem->getCameraPosition().y, cameraSystem->getCameraPosition().z);

		//ImGui::GetForegroundDrawList()->AddRect(mSceneContentMin, mSceneContentMax, 0xFFFF0000);
	}
	ImGui::End();
}

void SceneView::drawCameraSettingsPopup(PhysicsEngine::FreeLookCameraSystem* cameraSystem, bool* cameraSettingsActive)
{
	static bool cameraSettingsWindowOpen = false;

	ImGui::SetNextWindowSize(ImVec2(430, 450), ImGuiCond_FirstUseEver);
	if (ImGui::Begin("Editor Camera Settings", cameraSettingsActive, ImGuiWindowFlags_NoResize))
	{
		// Editor camera transform
		PhysicsEngine::Transform* transform = cameraSystem->getCamera()->getComponent<PhysicsEngine::Transform>();
		glm::vec3 position = transform->getPosition();
		glm::quat rotation = transform->getRotation();
		glm::vec3 scale = transform->getScale();
		glm::vec3 eulerAngles = glm::degrees(glm::eulerAngles(rotation));

		if (ImGui::InputFloat3("Position", glm::value_ptr(position)))
		{
			transform->setPosition(position);
		}

		if (ImGui::InputFloat3("Rotation", glm::value_ptr(eulerAngles)))
		{
			glm::quat x = glm::angleAxis(glm::radians(eulerAngles.x), glm::vec3(1.0f, 0.0f, 0.0f));
			glm::quat y = glm::angleAxis(glm::radians(eulerAngles.y), glm::vec3(0.0f, 1.0f, 0.0f));
			glm::quat z = glm::angleAxis(glm::radians(eulerAngles.z), glm::vec3(0.0f, 0.0f, 1.0f));

			transform->setRotation(z * y * x);
		}
		if (ImGui::InputFloat3("Scale", glm::value_ptr(scale)))
		{
			transform->setScale(scale);
		}

		PhysicsEngine::Viewport viewport = cameraSystem->getViewport();
		PhysicsEngine::Frustum frustum = cameraSystem->getFrustum();

		// Viewport settings
		if (ImGui::InputInt("X", &viewport.mX)) {
			cameraSystem->setViewport(viewport);
		}
		if (ImGui::InputInt("Y", &viewport.mY)) {
			cameraSystem->setViewport(viewport);
		}
		if (ImGui::InputInt("Width", &viewport.mWidth)) {
			cameraSystem->setViewport(viewport);
		}
		if (ImGui::InputInt("Height", &viewport.mHeight)) {
			cameraSystem->setViewport(viewport);
		}

		// Frustum settings
		if (ImGui::InputFloat("FOV", &frustum.mFov))
		{
			cameraSystem->setFrustum(frustum);
		}
		if (ImGui::InputFloat("Aspect Ratio", &frustum.mAspectRatio))
		{
			cameraSystem->setFrustum(frustum);
		}
		if (ImGui::InputFloat("Near Plane", &frustum.mNearPlane))
		{
			cameraSystem->setFrustum(frustum);
		}
		if (ImGui::InputFloat("Far Plane", &frustum.mFarPlane))
		{
			cameraSystem->setFrustum(frustum);
		}

		// SSAO and render path
		int renderPath = static_cast<int>(cameraSystem->getRenderPath());
		int ssao = static_cast<int>(cameraSystem->getSSAO());

		const char* renderPathNames[] = { "Forward", "Deferred" };
		const char* ssaoNames[] = { "On", "Off" };

		if (ImGui::Combo("Render Path", &renderPath, renderPathNames, 2))
		{
			cameraSystem->setRenderPath(static_cast<PhysicsEngine::RenderPath>(renderPath));
		}

		if (ImGui::Combo("SSAO", &ssao, ssaoNames, 2))
		{
			cameraSystem->setSSAO(static_cast<PhysicsEngine::CameraSSAO>(ssao));
		}

		// Directional light cascade splits
		int cascadeType = static_cast<int>(cameraSystem->getCamera()->mShadowCascades);

		const char* cascadeTypeNames[] = { "No Cascades", "Two Cascades", "Three Cascades", "Four Cascades", "Five Cascades" };

		if (ImGui::Combo("Shadow Cascades", &cascadeType, cascadeTypeNames, 5))
		{
			cameraSystem->getCamera()->mShadowCascades = static_cast<PhysicsEngine::ShadowCascades>(cascadeType);
		}

		if (cameraSystem->getCamera()->mShadowCascades != PhysicsEngine::ShadowCascades::NoCascades)
		{
			ImColor colors[5] = { ImColor(1.0f, 0.0f, 0.0f),
							  ImColor(0.0f, 1.0f, 0.0f),
							  ImColor(0.0f, 0.0f, 1.0f),
							  ImColor(0.0f, 1.0f, 1.0f),
							  ImColor(0.6f, 0.0f, 0.6f) };

			std::array<int, 5> splits = cameraSystem->getCamera()->getCascadeSplits();
			for (size_t i = 0; i < splits.size(); i++)
			{
				ImGui::PushItemWidth(0.125f * ImGui::GetWindowSize().x);

				ImGuiInputTextFlags flags = ImGuiInputTextFlags_None;

				if (i <= static_cast<int>(cameraSystem->getCamera()->mShadowCascades))
				{
					ImGui::PushStyleColor(ImGuiCol_FrameBg, (ImVec4)colors[i]);
					ImGui::PushStyleColor(ImGuiCol_FrameBgHovered, (ImVec4)colors[i]);
					ImGui::PushStyleColor(ImGuiCol_FrameBgActive, (ImVec4)colors[i]);
				}
				else
				{
					ImGui::PushStyleColor(ImGuiCol_FrameBg, (ImVec4)ImColor(0.5f, 0.5f, 0.5f));
					ImGui::PushStyleColor(ImGuiCol_FrameBgHovered, (ImVec4)ImColor(0.5f, 0.5f, 0.5f));
					ImGui::PushStyleColor(ImGuiCol_FrameBgActive, (ImVec4)ImColor(0.5f, 0.5f, 0.5f));

					flags |= ImGuiInputTextFlags_ReadOnly;
				}

				if (ImGui::InputInt(("##Cascade Splits" + std::to_string(i)).c_str(), &splits[i], 0, 100, flags))
				{
					cameraSystem->getCamera()->setCascadeSplit(i, splits[i]);
				}

				ImGui::PopStyleColor(3);
				ImGui::PopItemWidth();
				ImGui::SameLine();
			}
			ImGui::Text("Cascade Splits");
		}
	}

	ImGui::End();
}

ImVec2 SceneView::getWindowPos() const
{
	return mWindowPos;
}

ImVec2 SceneView::getContentMin() const
{
	return mContentMin;
}

ImVec2 SceneView::getContentMax() const
{
	return mContentMax;
}

float SceneView::getWindowWidth() const
{
	return mWindowWidth;
}

float SceneView::getWindowHeight() const
{
	return mWindowHeight;
}

bool SceneView::isOpen() const
{
	return mOpen;
}

bool SceneView::isFocused() const
{
	return mFocused;
}

bool SceneView::isHovered() const
{
	return mHovered;
}

bool SceneView::hoveredThisFrame() const
{
	return !mHoveredLastFrame && mHovered;
}

bool SceneView::unhoveredThisFrame() const
{
	return mHoveredLastFrame && !mHovered;
}