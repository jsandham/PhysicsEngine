#ifndef __EDITOR_UI_H__
#define __EDITOR_UI_H__

#include "core/Guid.h"

namespace PhysicsEditor
{
enum class InteractionType
{
    None,
    Entity,
    Texture2D,
    Texture3D,
    Cubemap,
    Shader,
    Material,
    Mesh,
    Font,
    CodeFile,
    Other
};

class EditorClipboard
{
  private:
    InteractionType draggedType;
    InteractionType selectedType;
    PhysicsEngine::Guid draggedId;
    PhysicsEngine::Guid selectedId;

  public:
    EditorClipboard();
    ~EditorClipboard();

    InteractionType getDraggedType() const;
    InteractionType getSelectedType() const;
    PhysicsEngine::Guid getDraggedId() const;
    PhysicsEngine::Guid getSelectedId() const;
    void setDraggedItem(InteractionType type, PhysicsEngine::Guid id);
    void setSelectedItem(InteractionType type, PhysicsEngine::Guid id);
    void clearDraggedItem();
    void clearSelectedItem();
};
} // namespace PhysicsEditor

#endif
