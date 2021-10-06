#ifndef APPLICATION_GRAPHICS_CONTEXT_H__
#define APPLICATION_GRAPHICS_CONTEXT_H__

#include "ApplicationWindow.h"

namespace PhysicsEngine
{
	class ApplicationGraphicsContext
	{
	public:
		ApplicationGraphicsContext();
		virtual ~ApplicationGraphicsContext() = 0;

		virtual void update() = 0;

		static ApplicationGraphicsContext* createApplicationGraphicsContext(void* window);
	};
}

#endif