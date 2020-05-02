#ifndef __TIME_H__
#define __TIME_H__

namespace PhysicsEngine
{
	struct Time
	{
		size_t frameCount;
		float time;
		float deltaTime;
		size_t deltaCycles;
	};

	float getFPS(Time time);
}


#endif