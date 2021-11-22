#ifndef __MAIN_MENU_BAR_H__
#define __MAIN_MENU_BAR_H__

#include <string>

#include "EditorClipboard.h"
#include "Filebrowser.h"

#include "views/ProjectWindow.h"
#include "views/PreferencesWindow.h"
#include "views/AboutPopup.h"
#include "views/BuildWindow.h"

namespace PhysicsEditor
{
class MenuBar
{
  private:
    // File
    bool mNewSceneClicked;
    bool mOpenSceneClicked;
    bool mSaveClicked;
    bool mSaveAsClicked;
    bool mNewProjectClicked;
    bool mOpenProjectClicked;
    bool mSaveProjectClicked;
    bool mBuildClicked;
    bool mQuitClicked;

    // Edit
    bool mPreferencesClicked;
    bool mPopulateTestScene;

    // Windows
    bool mOpenInspectorClicked;
    bool mOpenHierarchyClicked;
    bool mOpenConsoleClicked;
    bool mOpenSceneViewClicked;
    bool mOpenProjectViewClicked;

    // About
    bool mAboutClicked;

    Filebrowser mFilebrowser;
    
    ProjectWindow mProjectWindow;
    BuildWindow mBuildWindow;
    PreferencesWindow mPreferencesWindow;
    AboutPopup mAboutPopup;

  public:
    MenuBar();
    ~MenuBar();

    void init(Clipboard& clipboard);
    void update(Clipboard &clipboard);

    bool isNewSceneClicked() const;
    bool isOpenSceneClicked() const;
    bool isSaveClicked() const;
    bool isSaveAsClicked() const;
    bool isBuildClicked() const;
    bool isQuitClicked() const;
    bool isNewProjectClicked() const;
    bool isOpenProjectClicked() const;
    bool isSaveProjectClicked() const;
    bool isOpenInspectorCalled() const;
    bool isOpenHierarchyCalled() const;
    bool isOpenConsoleCalled() const;
    bool isOpenSceneViewCalled() const;
    bool isOpenProjectViewCalled() const;
    bool isAboutClicked() const;
    bool isPreferencesClicked() const;
    bool isPopulateTestSceneClicked() const;

  private:
    void showMenuFile(const Clipboard& clipboard);
    void showMenuEdit(const Clipboard& clipboard);
    void showMenuWindow(const Clipboard& clipboard);
    void showMenuHelp(const Clipboard& clipboard);
};
} // namespace PhysicsEditor

#endif