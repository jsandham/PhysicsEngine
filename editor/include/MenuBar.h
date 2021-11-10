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
    bool newSceneClicked;
    bool openSceneClicked;
    bool saveClicked;
    bool saveAsClicked;
    bool newProjectClicked;
    bool openProjectClicked;
    bool saveProjectClicked;
    bool buildClicked;
    bool quitClicked;

    // Edit
    bool preferencesClicked;
    bool runTestsClicked;

    // Windows
    bool openInspectorClicked;
    bool openHierarchyClicked;
    bool openConsoleClicked;
    bool openSceneViewClicked;
    bool openProjectViewClicked;

    // About
    bool aboutClicked;

    Filebrowser filebrowser;
    
    ProjectWindow projectWindow;
    BuildWindow buildWindow;
    PreferencesWindow preferencesWindow;
    AboutPopup aboutPopup;

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
    bool isRunTestsClicked() const;

  private:
    void showMenuFile(const Clipboard& clipboard);
    void showMenuEdit(const Clipboard& clipboard);
    void showMenuWindow(const Clipboard& clipboard);
    void showMenuHelp(const Clipboard& clipboard);

    void newScene(Clipboard& clipboard);
    void openScene(Clipboard& clipboard, const std::string& name, const std::filesystem::path& path);
    void saveScene(Clipboard& clipboard, const std::string& name, const std::filesystem::path& path);
    void newProject(Clipboard& clipboard);
    void openProject(Clipboard& clipboard);
    void saveProject(Clipboard& clipboard);
    void build(Clipboard& clipboard);
};
} // namespace PhysicsEditor

#endif