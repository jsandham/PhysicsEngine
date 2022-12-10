#ifndef RENDER_CONTEXT_H__
#define RENDER_CONTEXT_H__

namespace PhysicsEngine
{
	enum class RenderAPI
	{
		OpenGL,
		DirectX
	};

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