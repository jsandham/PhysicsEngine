#ifndef EDITOR_LAYER_H__
#define EDITOR_LAYER_H__

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
    class EditorLayer : public PhysicsEngine::Layer
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
        EditorLayer();
        ~EditorLayer();
        EditorLayer(const EditorLayer& other) = delete;
        EditorLayer& operator=(const EditorLayer& other) = delete;

        void init() override;
        void begin() override;
        void update(const PhysicsEngine::Time& time) override;
        void end() override;
        bool quit() override;
    };
} // namespace PhysicsEditor

#endif