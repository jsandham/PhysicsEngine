#ifndef __EDITOR_UI_H__
#define __EDITOR_UI_H__

#include <set>
#include <string>

#include "core/Guid.h"
#include "core/World.h"

#include "LibraryDirectory.h"
#include "EditorCameraSystem.h"

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

class Clipboard
{
    private:
        // project
        std::string mProjectName;
        std::string mProjectPath;

        // scene
        std::string mSceneName;
        std::string mScenePath;
        PhysicsEngine::Guid mSceneId;

        // editor world
        PhysicsEngine::World mWorld;

        // editor world systems
        PhysicsEngine::EditorCameraSystem* mEditorCameraSystem;
        PhysicsEngine::RenderSystem* mRenderSystem;
        PhysicsEngine::GizmoSystem* mGizmoSystem;
        PhysicsEngine::CleanUpSystem* mCleanUpSystem;

        std::pair<InteractionType, PhysicsEngine::Guid> mDragged;
        std::pair<InteractionType, PhysicsEngine::Guid> mSelected;

        LibraryDirectory mLibrary;

    public:
        bool mProjectDirty;
        bool mSceneDirty;

    public:
        Clipboard();
        ~Clipboard();
        Clipboard(const Clipboard& other) = delete;
        Clipboard& operator=(const Clipboard& other) = delete;

        void setActiveProject(const std::string& name, const std::string& path);
        void setActiveScene(const std::string& name, const std::string& path, const PhysicsEngine::Guid& sceneId);

        std::string getProjectPath() const;
        std::string getProjectName() const;
        std::string getScenePath() const;
        std::string getSceneName() const;
        PhysicsEngine::Guid getSceneId() const;
        bool isProjectDirty() const;
        bool isSceneDirty() const;

        PhysicsEngine::World* getWorld();

        LibraryDirectory& getLibrary();

        InteractionType getDraggedType() const;
        InteractionType getSelectedType() const;
        PhysicsEngine::Guid getDraggedId() const;
        PhysicsEngine::Guid getSelectedId() const;

        void setDraggedItem(InteractionType type, PhysicsEngine::Guid id);
        void setSelectedItem(InteractionType type, PhysicsEngine::Guid id);
        void clearDraggedItem();
        void clearSelectedItem();

        //bool isEntitySelected() const;
        //bool isMeshSelected() const;
        //bool isMaterialSelected() const;
        //bool isShaderSelected() const;
        //bool isTexture2DSelected() const;


};


















//struct EditorScene
//{
//    std::string name;
//    std::string path;
//    PhysicsEngine::Guid sceneId;
//    bool isDirty;
//};
//
//struct EditorProject
//{
//    std::string name;
//    std::string path;
//    bool isDirty;
//};
//
//class EditorClipboard
//{
//  private:
//    InteractionType draggedType;
//    InteractionType selectedType;
//    PhysicsEngine::Guid draggedId;
//    PhysicsEngine::Guid selectedId;
//
//    PhysicsEngine::World world;
//    EditorScene scene;
//    EditorProject project;
//    LibraryDirectory library;
//
//    std::set<PhysicsEngine::Guid> editorOnlyEntityIds;
//
//  public:
//    bool isDirty;
//
//  public:
//    EditorClipboard();
//    ~EditorClipboard();
//
//    InteractionType getDraggedType() const;
//    InteractionType getSelectedType() const;
//    PhysicsEngine::Guid getDraggedId() const;
//    PhysicsEngine::Guid getSelectedId() const;
//    void setDraggedItem(InteractionType type, PhysicsEngine::Guid id);
//    void setSelectedItem(InteractionType type, PhysicsEngine::Guid id);
//    void clearDraggedItem();
//    void clearSelectedItem();
//
//    std::string getScene() const;
//    std::string getProject() const;
//    std::string getScenePath() const;
//    std::string getProjectPath() const;
//
//    LibraryDirectory &getLibrary();
//
//    PhysicsEngine::World *getWorld();
//
//    std::set<PhysicsEngine::Guid> &getEditorOnlyIds();
//
//    void init();
//    void openScene(const std::string &name, const std::string &path);
//    void openProject(const std::string &name, const std::string &path);
//};
} // namespace PhysicsEditor

#endif
