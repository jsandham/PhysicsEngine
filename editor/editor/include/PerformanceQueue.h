#ifndef __PERFORMANCE_QUEUE_H__
#define __PERFORMANCE_QUEUE_H__

#include <vector>

namespace PhysicsEditor
{
	class PerformanceQueue
	{
		private:
			int numberOfSamples;
			int index;
			std::vector<float> queue;
			std::vector<float> data;

		public:
			PerformanceQueue();
			~PerformanceQueue();

			void setNumberOfSamples(int numberOfSamples);
			void addSample(float value);
			void clear();
			std::vector<float> getData();

	};
}

#endif
