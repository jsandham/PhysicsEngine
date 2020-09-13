#ifndef __POLYNOMIAL_ROOTS_H__
#define __POLYNOMIAL_ROOTS_H__

#include <vector>

namespace PhysicsEngine
{
	class PolynomialRoots
	{
		public:
			static std::vector<float> solveLinear(float a1, float a0);
			static std::vector<float> solveQuadratic(float a2, float a1, float a0);
			static std::vector<float> solveCubic(float a3, float a2, float a1, float a0);
			static std::vector<float> solveQuartic(float a4, float a3, float a2, float a1, float a0);
	};
}

#endif