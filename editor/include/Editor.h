#ifndef __EDITOR_H__
#define __EDITOR_H__

#include <core/Layer.h>
#include <core/Time.h>

#include "EditorClipboard.h"

#include "MenuBar.h"
#include "../include/views/Inspector.h"
#include "../include/views/Hierarchy.h"
#include "../include/views/SceneView.h"
#include "../include/views/ProjectView.h"
#include "../include/views/Console.h"
#include "../include/views/DebugOverlay.h"

namespace PhysicsEditor
{
    class Editor : public PhysicsEngine::Layer
    {
    private:
        Clipboard mClipboard;

        MenuBar mMenuBar;
        Inspector mInspector;
        Hierarchy mHierarchy;
        SceneView mSceneView;
        ProjectView mProjectView;
        Console mConsole;
        DebugOverlay mDebugOverlay;

    public:
        Editor();
        ~Editor();
        Editor(const Editor& other) = delete;
        Editor& operator=(const Editor& other) = delete;

        void init() override;
        void begin() override;
        void update(const PhysicsEngine::Time& time) override;
        void end() override;
    };
} // namespace PhysicsEditor

#endif