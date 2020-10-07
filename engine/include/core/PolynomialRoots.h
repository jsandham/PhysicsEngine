#ifndef __POLYNOMIAL_ROOTS_H__
#define __POLYNOMIAL_ROOTS_H__

#include <vector>

namespace PhysicsEngine
{
class PolynomialRoots
{
  private:
    static std::vector<float> solveDepressedQuadratic(float p);
    static std::vector<float> solveDepressedCubic(float p, float q);
    static std::vector<float> solveDepressedQuartic(float p, float q, float r);

  public:
    static std::vector<float> solveLinear(float a1, float a0);
    static std::vector<float> solveQuadratic(float a4, float a2, float a0);
    static std::vector<float> solveBiQuadratic(float a4, float a2, float a0);
    static std::vector<float> solveCubic(float a3, float a2, float a1, float a0);
    static std::vector<float> solveQuartic(float a4, float a3, float a2, float a1, float a0);
};
} // namespace PhysicsEngine

#endif