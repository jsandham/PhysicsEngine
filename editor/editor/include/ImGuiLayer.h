#ifndef IMGUI_LAYER_H__
#define IMGUI_LAYER_H__

#include <core/Layer.h>

namespace PhysicsEditor
{
	class ImGuiLayer : public PhysicsEngine::Layer
	{
	public:
        ImGuiLayer();
        ~ImGuiLayer();
        ImGuiLayer(const ImGuiLayer& other) = delete;
        ImGuiLayer& operator=(const ImGuiLayer& other) = delete;

        void init() override;
        void begin() override;
        void update() override;
        void end() override;
	};
}

#endif
