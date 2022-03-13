#ifndef LAYER_H__
#define LAYER_H__

#include <string>

#include "Time.h"

namespace PhysicsEngine
{
	class Layer
	{
	private:
		std::string mName;

	public:
		Layer(const std::string& name = "Layer");
		virtual ~Layer() = 0;

		virtual void init() = 0;
		virtual void begin() = 0;
		virtual void update(const Time& time) = 0;
		virtual void end() = 0;
        virtual bool quit() = 0;
	};
}

#endif