#ifndef __BUILD_WINDOW_H__
#define __BUILD_WINDOW_H__

#include <thread>
#include <vector>
#include <assert.h>

#include "PopupWindow.h"
#include "../Filebrowser.h"

namespace PhysicsEditor
{
    enum class TargetPlatform
    {
        Windows = 0,
        Linux = 1
    };

class BuildWindow : public PopupWindow
{
private:
    TargetPlatform mTargetPlatform;
    Filebrowser mFilebrowser;
     
    float mBuildCompletion;
    std::string mBuildStep;

    std::atomic<bool> mLaunchBuild{ false };
    std::atomic<bool> mBuildInProgress{ false };
    std::thread mWorker;

  public:
    BuildWindow();
    ~BuildWindow();
    BuildWindow(const BuildWindow &other) = delete;
    BuildWindow &operator=(const BuildWindow &other) = delete;

    void init(Clipboard &clipboard);
    void update(Clipboard &clipboard);

  private:
    void build();
    void doWork();

};
} // namespace PhysicsEditor

#endif
