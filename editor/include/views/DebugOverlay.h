#ifndef DEBUG_OVERLAY_H__
#define DEBUG_OVERLAY_H__

#include "../EditorClipboard.h"
#include "../PerformanceQueue.h"

namespace PhysicsEditor
{
	class DebugOverlay
	{
	private:
		float mMaxFPS;
		PerformanceQueue mPerfQueue;

	public:
		DebugOverlay();
		~DebugOverlay();
		DebugOverlay(const DebugOverlay& other) = delete;
		DebugOverlay& operator=(const DebugOverlay& other) = delete;

		void init(Clipboard& clipboard);
		void update(Clipboard& clipboard);

	private:
		void sceneTab(Clipboard& clipboard);
		void shaderTab(Clipboard& clipboard);
	};
} // namespace PhysicsEditor

#endif