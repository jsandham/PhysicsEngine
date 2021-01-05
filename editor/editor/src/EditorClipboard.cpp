#include "../include/EditorClipboard.h"

#include "../include/EditorOnlyEntityCreation.h"

#include "../include/EditorCameraSystem.h"

#include "systems/CleanUpSystem.h"
#include "systems/GizmoSystem.h"
#include "systems/RenderSystem.h"

using namespace PhysicsEditor;

EditorClipboard::EditorClipboard()
{
    draggedType = InteractionType::None;
    selectedType = InteractionType::None;
    draggedId = PhysicsEngine::Guid::INVALID;
    selectedId = PhysicsEngine::Guid::INVALID;
}

EditorClipboard::~EditorClipboard()
{
}

InteractionType EditorClipboard::getDraggedType() const
{
    return draggedType;
}

InteractionType EditorClipboard::getSelectedType() const
{
    return selectedType;
}

PhysicsEngine::Guid EditorClipboard::getDraggedId() const
{
    return draggedId;
}

PhysicsEngine::Guid EditorClipboard::getSelectedId() const
{
    return selectedId;
}

void EditorClipboard::setDraggedItem(InteractionType type, PhysicsEngine::Guid id)
{
    draggedType = type;
    draggedId = id;
}

void EditorClipboard::setSelectedItem(InteractionType type, PhysicsEngine::Guid id)
{
    selectedType = type;
    selectedId = id;
}

void EditorClipboard::clearDraggedItem()
{
    draggedType = InteractionType::None;
    draggedId = PhysicsEngine::Guid::INVALID;
}

void EditorClipboard::clearSelectedItem()
{
    selectedType = InteractionType::None;
    selectedId = PhysicsEngine::Guid::INVALID;
}

std::string EditorClipboard::getScene() const
{
    return scene.name;
}

std::string EditorClipboard::getProject() const
{
    return project.name;
}

std::string EditorClipboard::getScenePath() const
{
    return scene.path;
}

std::string EditorClipboard::getProjectPath() const
{
    return project.path;
}

PhysicsEngine::World *EditorClipboard::getWorld()
{
    return &world;
}

LibraryDirectory &EditorClipboard::getLibrary()
{
    return library;
}

std::set<PhysicsEngine::Guid> &EditorClipboard::getEditorOnlyIds()
{
    return editorOnlyEntityIds;
}

void EditorClipboard::openScene(const std::string &name, const std::string &path)
{
    scene.name = name;
    scene.path = path;
}

void EditorClipboard::openScene(const std::string &name, const std::string &path, const std::string &metaPath,
                                const std::string &libraryPath, const PhysicsEngine::Guid &sceneId)
{
    scene.name = name;
    scene.path = path;
    scene.metaPath = metaPath;
    scene.libraryPath = libraryPath;
    scene.sceneId = sceneId;
}

void EditorClipboard::openProject(const std::string &name, const std::string &path)
{
    project.name = name;
    project.path = path;
}

void EditorClipboard::init()
{
    // add editor camera to world
    PhysicsEditor::createEditorCamera(&world, editorOnlyEntityIds);

    // add camera, render, and cleanup system to world
    world.addSystem<PhysicsEngine::EditorCameraSystem>(0);
    // add simple editor render pass system to render line floor and default skymap
    world.addSystem<PhysicsEngine::RenderSystem>(1);
    // add gizmo system
    world.addSystem<PhysicsEngine::GizmoSystem>(2);
    // add simple editor render system to render gizmo's
    world.addSystem<PhysicsEngine::CleanUpSystem>(3);

    PhysicsEngine::RenderSystem *renderSystem = world.getSystem<PhysicsEngine::RenderSystem>();

    renderSystem->mRenderToScreen = false;
}
