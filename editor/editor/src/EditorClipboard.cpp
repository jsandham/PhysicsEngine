#include "../include/EditorClipboard.h"

using namespace PhysicsEditor;

EditorClipboard::EditorClipboard()
{
    draggedType = InteractionType::None;
    draggedId = PhysicsEngine::Guid::INVALID;
    selectedType = InteractionType::None;
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
