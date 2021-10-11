#ifndef RENDERER_API_H__
#define RENDERER_API_H__

namespace PhysicsEngine
{
	class RendererAPI
	{
	public:
		RendererAPI();
		virtual ~RendererAPI() = 0;

		virtual void init(void* window) = 0;
		virtual void update() = 0;
		virtual void cleanup() = 0;

		static RendererAPI* createRendererAPI();
	};
}

#endif