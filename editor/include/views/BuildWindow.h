#ifndef __BUILD_WINDOW_H__
#define __BUILD_WINDOW_H__

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

  public:
    BuildWindow();
    ~BuildWindow();
    BuildWindow(const BuildWindow &other) = delete;
    BuildWindow &operator=(const BuildWindow &other) = delete;

    void init(Clipboard &clipboard);
    void update(Clipboard &clipboard);

    void build(const std::filesystem::path& path);
};
} // namespace PhysicsEditor

#endif
