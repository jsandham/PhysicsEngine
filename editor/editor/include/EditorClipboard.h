#ifndef __EDITOR_UI_H__
#define __EDITOR_UI_H__

#define GLM_FORCE_RADIANS

#include <set>
#include <string>
#include <filesystem>

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
    public:
        std::string mProjectName;
        std::string mSceneName;
        std::filesystem::path mProjectPath;
        std::filesystem::path mScenePath;
        std::filesystem::path mSelectedPath;
        std::filesystem::path mDraggedPath;
        PhysicsEngine::Guid mSceneId;
        PhysicsEngine::Guid mSelectedId;
        PhysicsEngine::Guid mDraggedId;
        InteractionType mSelectedType;
        InteractionType mDraggedType;
        PhysicsEngine::Guid mSceneViewTempEntityId;
        PhysicsEngine::Entity* mSceneViewTempEntity;
        PhysicsEngine::Transform* mSceneViewTempTransform;

        PhysicsEngine::World mWorld;
        PhysicsEngine::EditorCameraSystem* mEditorCameraSystem;
        PhysicsEngine::RenderSystem* mRenderSystem;
        PhysicsEngine::GizmoSystem* mGizmoSystem;
        PhysicsEngine::CleanUpSystem* mCleanUpSystem;

        LibraryDirectory mLibrary;

        bool mProjectDirty;
        bool mSceneDirty;

        bool mInspectorOpen;
        bool mInspectorHovered;
        bool mInspectorFocused;
        bool mInspectorOpenedThisFrame;
        bool mInspectorHoveredThisFrame;
        bool mInspectorFocusedThisFrame;
        bool mInspectorClosedThisFrame;
        bool mInspectorUnhoveredThisFrame;
        bool mInspectorUnfocusedThisFrame;

        bool mSceneViewOpen;
        bool mSceneViewHovered;
        bool mSceneViewFocused;
        bool mSceneViewOpenedThisFrame;
        bool mSceneViewHoveredThisFrame;
        bool mSceneViewFocusedThisFrame;
        bool mSceneViewClosedThisFrame;
        bool mSceneViewUnhoveredThisFrame;
        bool mSceneViewUnfocusedThisFrame;

        bool mHierarchyOpen;
        bool mHierarchyHovered;
        bool mHierarchyFocused;
        bool mHierarchyOpenedThisFrame;
        bool mHierarchyHoveredThisFrame;
        bool mHierarchyFocusedThisFrame;
        bool mHierarchyClosedThisFrame;
        bool mHierarchyUnhoveredThisFrame;
        bool mHierarchyUnfocusedThisFrame;

        bool mProjectViewOpen;
        bool mProjectViewHovered;
        bool mProjectViewFocused;
        bool mProjectViewOpenedThisFrame;
        bool mProjectViewHoveredThisFrame;
        bool mProjectViewFocusedThisFrame;
        bool mProjectViewClosedThisFrame;
        bool mProjectViewUnhoveredThisFrame;
        bool mProjectViewUnfocusedThisFrame;

        bool mConsoleOpen;
        bool mConsoleHovered;
        bool mConsoleFocused;
        bool mConsoleOpenedThisFrame;
        bool mConsoleHoveredThisFrame;
        bool mConsoleFocusedThisFrame;
        bool mConsoleClosedThisFrame;
        bool mConsoleUnhoveredThisFrame;
        bool mConsoleUnfocusedThisFrame;

    public:
        Clipboard();
        ~Clipboard();
        Clipboard(const Clipboard& other) = delete;
        Clipboard& operator=(const Clipboard& other) = delete;

        void setActiveProject(const std::string& name, const std::string& path);
        void setActiveScene(const std::string& name, const std::string& path, const PhysicsEngine::Guid& sceneId);

        std::filesystem::path getProjectPath() const;
        std::string getProjectName() const;
        std::filesystem::path getScenePath() const;
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
        std::filesystem::path getSelectedPath() const;

        void setDraggedItem(InteractionType type, PhysicsEngine::Guid id);
        void setSelectedItem(InteractionType type, PhysicsEngine::Guid id);
        void setSelectedItem(InteractionType type, std::string path);
        void clearDraggedItem();
        void clearSelectedItem();
};

} // namespace PhysicsEditor

#endif
