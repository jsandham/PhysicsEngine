#ifndef EDITOR_CLIPBOARD_H__
#define EDITOR_CLIPBOARD_H__

#define GLM_FORCE_RADIANS

#include <set>
#include <string>
#include <filesystem>

#include "core/Guid.h"
#include "core/World.h"
#include <core/Time.h>
#include <systems/FreeLookCameraSystem.h>

#include "LibraryDirectory.h"

#include "ProjectTree.h"
#include "InteractionType.h"

namespace PhysicsEditor
{
    enum class View
    {
        Inspector,
        SceneView,
        Hierarchy,
        ProjectView,
        Console,
        Count
    };

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
        PhysicsEngine::FreeLookCameraSystem* mCameraSystem;
        PhysicsEngine::TerrainSystem* mTerrainSystem;
        PhysicsEngine::RenderSystem* mRenderSystem;
        PhysicsEngine::GizmoSystem* mGizmoSystem;
        PhysicsEngine::CleanUpSystem* mCleanUpSystem;

        LibraryDirectory mLibrary;

        bool mProjectDirty;
        bool mSceneDirty;

        bool mOpen[View::Count];
        bool mHovered[View::Count];
        bool mFocused[View::Count];
        bool mOpenedThisFrame[View::Count];
        bool mHoveredThisFrame[View::Count];
        bool mFocusedThisFrame[View::Count];
        bool mClosedThisFrame[View::Count];
        bool mUnhoveredThisFrame[View::Count];
        bool mUnfocusedThisFrame[View::Count];

        PhysicsEngine::Time mTime;

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
