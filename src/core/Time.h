#ifndef __TIME_H__
#define __TIME_H__

namespace PhysicsEngine
{
	class Time
	{
		public:
			static int frameCount;
			static int deltaCycles;
			static float time;
			static float deltaTime;

		private:
			Time();
	};
}

#endif