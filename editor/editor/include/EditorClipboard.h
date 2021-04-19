#ifndef __EDITOR_UI_H__
#define __EDITOR_UI_H__

#include <set>
#include <string>

#include "core/Guid.h"
#include "core/World.h"

#include "LibraryDirectory.h"
#include "EditorCameraSystem.h"

#include "ProjectTree.h"
#include "InteractionType.h"

namespace PhysicsEditor
{
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

        LibraryDirectory mLibrary;

    public:
        bool mInspectorHovered;
        bool mSceneViewHovered;
        bool mHierarchyHovered;
        bool mProjectViewHovered;
        bool mConsoleHovered;

    public:
        InteractionType mSelectedType;
        //InteractionType mSelectedFileType;
        PhysicsEngine::Guid mSelectedId;
        std::string mSelectedPath;

        InteractionType mDraggedType;
        //InteractionType mDraggedFileType;
        PhysicsEngine::Guid mDraggedId;
        std::string mDraggedPath;

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
        std::string getSelectedPath() const;

        void setDraggedItem(InteractionType type, PhysicsEngine::Guid id);
        void setSelectedItem(InteractionType type, PhysicsEngine::Guid id);
        void setSelectedItem(InteractionType type, std::string path);
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
