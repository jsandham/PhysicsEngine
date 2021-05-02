#ifndef __EDITOR_H__
#define __EDITOR_H__

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
    class Editor
    {
    private:
        Clipboard mClipboard;

        //CommandManager mCommand;
        //Undo mUndo;

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

        void init();
        void update();
    };
} // namespace PhysicsEditor

#endif