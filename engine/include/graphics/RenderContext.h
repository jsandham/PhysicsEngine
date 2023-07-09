#ifndef RENDER_CONTEXT_H__
#define RENDER_CONTEXT_H__

namespace PhysicsEngine
{
	enum class RenderAPI
	{
		OpenGL,
		DirectX
	};

	constexpr auto RenderAPIToString(RenderAPI api)
    {
        switch (api)
        {
        case RenderAPI::OpenGL:
            return "OpenGL";
        case RenderAPI::DirectX:
            return "DirectX";
        }
    }

	constexpr auto GetShaderLanguageStringFromRenderAPI(RenderAPI api)
    {
        switch (api)
        {
        case RenderAPI::OpenGL:
            return "GLSL";
        case RenderAPI::DirectX:
            return "HLSL";
        }
    }

	class RenderContext
	{
	public:
		static RenderAPI sAPI;
		static RenderContext* sContext;

		static RenderAPI getRenderAPI();
		static void setRenderAPI(RenderAPI api);
		static void createRenderContext(void* window);
	};
}

#endif