#ifndef __PERFORMANCE_GRAPH_H__
#define __PERFORMANCE_GRAPH_H__

#include <vector>

#include "../graphics/GLHandle.h"

namespace PhysicsEngine
{
	class PerformanceGraph
	{
		private:
			float x;
			float y;
			float width;
			float height;
			float rangeMin;
			float rangeMax;
			float currentSample;
			int numberOfSamples;

		public:
			std::vector<float> vertices;
			GLHandle graphVAO;
			GLHandle vertexVBO;

		public:
			PerformanceGraph(float x, float y, float width, float height, float rangeMin, float rangeMax, int numberOfSamples);
			~PerformanceGraph();

			void add(float sample);
	};
}
#endif