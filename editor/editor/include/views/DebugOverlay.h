#ifndef __DEBUG_OVERLAY_H__
#define __DEBUG_OVERLAY_H__

#include "Window.h"
#include "../PerformanceQueue.h"

namespace PhysicsEditor
{
    class DebugOverlay : public Window
    {
    private:
        float mMaxFPS;
        PerformanceQueue mPerfQueue;

    public:
        DebugOverlay();
        ~DebugOverlay();
        DebugOverlay(const DebugOverlay& other) = delete;
        DebugOverlay& operator=(const DebugOverlay& other) = delete;

        void init(Clipboard& clipboard) override;
        void update(Clipboard& clipboard) override;

        void sceneTab(Clipboard& clipboard);
        void shaderTab(Clipboard& clipboard);
    };
} // namespace PhysicsEditor

#endif