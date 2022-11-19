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

//#ifndef RENDER_CONTEXT_H__
//#define RENDER_CONTEXT_H__
//
//namespace PhysicsEngine
//{
//	enum class RenderAPI
//	{
//		OpenGL,
//		DirectX
//	};
//
//	class RenderContext
//	{
//	public:
//        static RenderAPI sAPI;
//
//        RenderContext();
//		virtual ~RenderContext() = 0;
//
//		virtual void init(void* window) = 0;
//		virtual void update() = 0;
//		virtual void cleanup() = 0;
//		virtual void turnVsyncOn() = 0;
//		virtual void turnVsyncOff() = 0;
//
//		static RenderAPI getRenderAPI();
//        static void setRenderAPI(RenderAPI api);
//		static RenderContext *createRenderContext();
//	};
//}
//
//#endif