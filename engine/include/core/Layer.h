#ifndef LAYER_H__
#define LAYER_H__

#include <string>

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
		virtual void update() = 0;
	};
}

#endif