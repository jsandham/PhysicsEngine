#ifndef IMGUI_LAYER_H__
#define IMGUI_LAYER_H__

#include <core/Layer.h>
#include <core/Time.h>

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
        void update(const PhysicsEngine::Time& time) override;
        void end() override;
        bool quit() override;
	};
}

#endif
