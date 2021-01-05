#ifndef __EDITOR_UI_H__
#define __EDITOR_UI_H__

#include <set>
#include <string>

#include "core/Guid.h"
#include "core/World.h"

#include "LibraryDirectory.h"

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

struct EditorScene
{
    std::string name;
    std::string path;
    std::string metaPath;
    std::string libraryPath;
    PhysicsEngine::Guid sceneId;
    bool isDirty;
};

struct EditorProject
{
    std::string name;
    std::string path;
    bool isDirty;
};

class EditorClipboard
{
  private:
    InteractionType draggedType;
    InteractionType selectedType;
    PhysicsEngine::Guid draggedId;
    PhysicsEngine::Guid selectedId;

    PhysicsEngine::World world;
    EditorScene scene;
    EditorProject project;
    LibraryDirectory library;

    std::set<PhysicsEngine::Guid> editorOnlyEntityIds;

  public:
    bool isDirty;

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

    std::string getScene() const;
    std::string getProject() const;
    std::string getScenePath() const;
    std::string getProjectPath() const;

    LibraryDirectory &getLibrary();

    PhysicsEngine::World *getWorld();

    std::set<PhysicsEngine::Guid> &getEditorOnlyIds();

    void init();
    void openScene(const std::string &name, const std::string &path);
    void openScene(const std::string &name, const std::string &path, const std::string &metaPath,
                   const std::string &libraryPath, const PhysicsEngine::Guid &sceneId);
    void openProject(const std::string &name, const std::string &path);
};
} // namespace PhysicsEditor

#endif
