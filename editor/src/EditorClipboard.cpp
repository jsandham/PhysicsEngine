#include "../include/EditorClipboard.h"

#include "systems/CleanUpSystem.h"
#include "systems/GizmoSystem.h"
#include "systems/RenderSystem.h"

using namespace PhysicsEditor;

Clipboard::Clipboard()
{
    mSelectedType = InteractionType::None;
    mSelectedId = PhysicsEngine::Guid::INVALID;
    mSelectedPath = std::filesystem::path();

    mDraggedType = InteractionType::None;
    mDraggedId = PhysicsEngine::Guid::INVALID;
    mDraggedPath = std::filesystem::path();

    mProjectName = "";
    mProjectPath = std::filesystem::path();
    mSceneName = "";
    mScenePath = std::filesystem::path();
    mSceneId = PhysicsEngine::Guid::INVALID;

    // add camera, render, and cleanup system to world
    mCameraSystem = mWorld.addSystem<PhysicsEngine::FreeLookCameraSystem>(0);
    // add camera, render, and cleanup system to world
    mTerrainSystem = mWorld.addSystem<PhysicsEngine::TerrainSystem>(1);
    // add simple editor render pass system to render line floor and default skymap
    mRenderSystem = mWorld.addSystem<PhysicsEngine::RenderSystem>(2);
    // add gizmo system
    mGizmoSystem = mWorld.addSystem<PhysicsEngine::GizmoSystem>(3);
    // add simple editor render system to render gizmo's
    mCleanUpSystem = mWorld.addSystem<PhysicsEngine::CleanUpSystem>(4);

    mCameraSystem->mHide = PhysicsEngine::HideFlag::DontSave;
    mTerrainSystem->mHide = PhysicsEngine::HideFlag::DontSave;
    mRenderSystem->mHide = PhysicsEngine::HideFlag::DontSave;
    mGizmoSystem->mHide = PhysicsEngine::HideFlag::DontSave;
    mCleanUpSystem->mHide = PhysicsEngine::HideFlag::DontSave;

    mProjectDirty = false;
    mSceneDirty = false;

    mInspectorHovered = false;
    mSceneViewHovered = false;
    mHierarchyHovered = false;
    mProjectViewHovered = false;
    mConsoleHovered = false;
}

Clipboard::~Clipboard()
{
}


void Clipboard::setActiveProject(const std::string& name, const std::string& path)
{
    mProjectName = name;
    mProjectPath = path;
}

void Clipboard::setActiveScene(const std::string& name, const std::string& path, const PhysicsEngine::Guid& sceneId)
{
    mSceneName = name;
    mScenePath = path;
    mSceneId = sceneId;
}

std::filesystem::path Clipboard::getProjectPath() const
{
    return mProjectPath;
}

std::string Clipboard::getProjectName() const
{
    return mProjectName;
}

std::filesystem::path Clipboard::getScenePath() const
{
    return mScenePath;
}

std::string Clipboard::getSceneName() const
{
    return mSceneName;
}

PhysicsEngine::Guid Clipboard::getSceneId() const
{
    return mSceneId;
}

bool Clipboard::isProjectDirty() const
{
    return mProjectDirty;
}

bool Clipboard::isSceneDirty() const
{
    return mSceneDirty;
}

PhysicsEngine::World* Clipboard::getWorld()
{
    return &mWorld;
}

LibraryDirectory& Clipboard::getLibrary()
{
    return mLibrary;
}

InteractionType Clipboard::getDraggedType() const
{
    return mDraggedType;
}

InteractionType Clipboard::getSelectedType() const
{
    return mSelectedType;
}

PhysicsEngine::Guid Clipboard::getDraggedId() const
{
    return mDraggedId;
}

PhysicsEngine::Guid Clipboard::getSelectedId() const
{
    return mSelectedId;
}

std::filesystem::path Clipboard::getSelectedPath() const
{
    return mSelectedPath;
}

void Clipboard::setDraggedItem(InteractionType type, PhysicsEngine::Guid id)
{
    mDraggedType = type;
    mDraggedId = id;
}

void Clipboard::setSelectedItem(InteractionType type, PhysicsEngine::Guid id)
{
    mSelectedType = type;
    mSelectedId = id;
}

void Clipboard::setSelectedItem(InteractionType type, std::string path)
{
    mSelectedType = type;
    mSelectedPath = path;
}

void Clipboard::clearDraggedItem()
{
    mDraggedType = InteractionType::None;
    mDraggedId = PhysicsEngine::Guid::INVALID;
    mDraggedPath = "";
}

void Clipboard::clearSelectedItem()
{
    mSelectedType = InteractionType::None;
    mSelectedId = PhysicsEngine::Guid::INVALID;
    mSelectedPath = "";
}