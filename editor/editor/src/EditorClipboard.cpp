#include "../include/EditorClipboard.h"

#include "../include/EditorOnlyEntityCreation.h"

#include "../include/EditorCameraSystem.h"

#include "systems/CleanUpSystem.h"
#include "systems/GizmoSystem.h"
#include "systems/RenderSystem.h"

using namespace PhysicsEditor;

Clipboard::Clipboard()
{
    mSelectedType = InteractionType::None;
    //mSelectedFileType = FileType::None;
    mSelectedId = PhysicsEngine::Guid::INVALID;
    mSelectedPath = "";

    mDraggedType = InteractionType::None;
    //mDraggedFileType = FileType::None;
    mDraggedId = PhysicsEngine::Guid::INVALID;
    mDraggedPath = "";

    mProjectName = "";
    mProjectPath = "";
    mSceneName = "";
    mScenePath = "";
    mSceneId = PhysicsEngine::Guid::INVALID;

    // add camera, render, and cleanup system to world
    mEditorCameraSystem = mWorld.addSystem<PhysicsEngine::EditorCameraSystem>(0);
    // add simple editor render pass system to render line floor and default skymap
    mRenderSystem = mWorld.addSystem<PhysicsEngine::RenderSystem>(1);
    // add gizmo system
    mGizmoSystem = mWorld.addSystem<PhysicsEngine::GizmoSystem>(2);
    // add simple editor render system to render gizmo's
    mCleanUpSystem = mWorld.addSystem<PhysicsEngine::CleanUpSystem>(3);

    mRenderSystem->mRenderToScreen = false;

    mEditorCameraSystem->mHide = PhysicsEngine::HideFlag::DontSave;
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

std::string Clipboard::getProjectPath() const
{
    return mProjectPath;
}

std::string Clipboard::getProjectName() const
{
    return mProjectName;
}

std::string Clipboard::getScenePath() const
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

std::string Clipboard::getSelectedPath() const
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
    //mDraggedFileType = FileType::None;
    mDraggedId = PhysicsEngine::Guid::INVALID;
    mDraggedPath = "";
}

void Clipboard::clearSelectedItem()
{
    mSelectedType = InteractionType::None;
    //mSelectedFileType = FileType::None;
    mSelectedId = PhysicsEngine::Guid::INVALID;
    mSelectedPath = "";
}
































//EditorClipboard::EditorClipboard()
//{
//    draggedType = InteractionType::None;
//    selectedType = InteractionType::None;
//    draggedId = PhysicsEngine::Guid::INVALID;
//    selectedId = PhysicsEngine::Guid::INVALID;
//}
//
//EditorClipboard::~EditorClipboard()
//{
//}
//
//InteractionType EditorClipboard::getDraggedType() const
//{
//    return draggedType;
//}
//
//InteractionType EditorClipboard::getSelectedType() const
//{
//    return selectedType;
//}
//
//PhysicsEngine::Guid EditorClipboard::getDraggedId() const
//{
//    return draggedId;
//}
//
//PhysicsEngine::Guid EditorClipboard::getSelectedId() const
//{
//    return selectedId;
//}
//
//void EditorClipboard::setDraggedItem(InteractionType type, PhysicsEngine::Guid id)
//{
//    draggedType = type;
//    draggedId = id;
//}
//
//void EditorClipboard::setSelectedItem(InteractionType type, PhysicsEngine::Guid id)
//{
//    selectedType = type;
//    selectedId = id;
//}
//
//void EditorClipboard::clearDraggedItem()
//{
//    draggedType = InteractionType::None;
//    draggedId = PhysicsEngine::Guid::INVALID;
//}
//
//void EditorClipboard::clearSelectedItem()
//{
//    selectedType = InteractionType::None;
//    selectedId = PhysicsEngine::Guid::INVALID;
//}
//
//std::string EditorClipboard::getScene() const
//{
//    return scene.name;
//}
//
//std::string EditorClipboard::getProject() const
//{
//    return project.name;
//}
//
//std::string EditorClipboard::getScenePath() const
//{
//    return scene.path;
//}
//
//std::string EditorClipboard::getProjectPath() const
//{
//    return project.path;
//}
//
//PhysicsEngine::World *EditorClipboard::getWorld()
//{
//    return &world;
//}
//
//LibraryDirectory &EditorClipboard::getLibrary()
//{
//    return library;
//}
//
//std::set<PhysicsEngine::Guid> &EditorClipboard::getEditorOnlyIds()
//{
//    return editorOnlyEntityIds;
//}
//
//void EditorClipboard::openScene(const std::string& name, const std::string& path)
//{
//    scene.name = name;
//    scene.path = path;
//    scene.sceneId = getWorld()->getSceneId();// sceneId;
//}
//
//void EditorClipboard::openProject(const std::string &name, const std::string &path)
//{
//    project.name = name;
//    project.path = path;
//}
//
//void EditorClipboard::init()
//{
//    // add editor camera to world
//    PhysicsEditor::createEditorCamera(&world, editorOnlyEntityIds);
//
//    // add camera, render, and cleanup system to world
//    world.addSystem<PhysicsEngine::EditorCameraSystem>(0);
//    // add simple editor render pass system to render line floor and default skymap
//    world.addSystem<PhysicsEngine::RenderSystem>(1);
//    // add gizmo system
//    world.addSystem<PhysicsEngine::GizmoSystem>(2);
//    // add simple editor render system to render gizmo's
//    world.addSystem<PhysicsEngine::CleanUpSystem>(3);
//
//    PhysicsEngine::RenderSystem *renderSystem = world.getSystem<PhysicsEngine::RenderSystem>();
//
//    renderSystem->mRenderToScreen = false;
//}
