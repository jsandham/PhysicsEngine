#include "../../include/core/PolynomialRoots.h"

#include <algorithm>

#include "../../include/glm/glm.hpp"
#include "../../include/glm/detail/func_trigonometric.hpp"
#include "../../include/glm/gtc/constants.hpp"

using namespace PhysicsEngine;

// Solve x^2 + p = 0
std::vector<float> PolynomialRoots::solveDepressedQuadratic(float p)
{
	std::vector<float> roots;

	float h = p;

	if (h > 0) {
		return roots;
	}
	else if (h < 0) {
		float temp = glm::sqrt(-p);
		roots.push_back(temp);
		roots.push_back(-temp);

		return roots;
	}
	else {
		roots.push_back(0.0f);
		roots.push_back(0.0f);

		return roots;
	}
}

// Solve t^3 + p * t + q = 0
std::vector<float> PolynomialRoots::solveDepressedCubic(float p, float q)
{
	std::vector<float> roots;

	const float oneThird = 1.0f / 3.0f;

	if (q == 0) {
		// solve depressed quadratic
		roots = solveDepressedQuadratic(p);

		// add t==0 root
		roots.push_back(0.0f);
		return roots;
	}

	if (p == 0) {
		float t0 = pow(abs(q), oneThird);

		if (q > 0.0f) {
			t0 *= -1.0f;
		}

		roots.push_back(t0);
		return roots;
	}

	// Discriminant
	float h = 27 * q * q + 4 * p * p * p;

	if (h > 0.0f) {
		// One real root
		float sqrth = glm::sqrt(h / 108);
		float R = -0.5f * q + sqrth;
		float T = -0.5f * q - sqrth;

		if (T >= 0.0) {
			T = pow(T, oneThird);
		}
		else {
			T = -pow(-T, oneThird);
		}

		if (R >= 0.0) {
			R = pow(R, oneThird);
		}
		else {
			R = -pow(-R, oneThird);
		}

		float t0 = T + R;

		roots.push_back(t0);

		return roots;
	}
	else if (h < 0.0f) {
		// Three real roots
		float R = 2 * glm::sqrt(-p / 3);
		float S = 1.5f * q * glm::sqrt(-3 / p) / p;
		float T = glm::acos(S) / 3.0f;

		float t0 = R * glm::cos(T);
		float t1 = R * glm::cos(T - 2 * glm::pi<float>() / 3.0f);
		float t2 = R * glm::cos(T - 4 * glm::pi<float>() / 3.0f);

		roots.push_back(t0);
		roots.push_back(t1);
		roots.push_back(t2);

		return roots;
	}
	else {
		// One real root and one real double root
		float t0 = 3 * q / p;
		float t1 = -1.5f * q / p;
		float t2 = t1;

		roots.push_back(t0);
		roots.push_back(t1);
		roots.push_back(t2);

		return roots;
	}
}

// Solve t^4 + p*t^2 + q*t + r = 0
std::vector<float> PolynomialRoots::solveDepressedQuartic(float p, float q, float r)
{
	std::vector<float> roots;

	if (r == 0.0f) {
		// solve depressed cubic
		roots = solveDepressedCubic(p, q);

		// add t==0 root
		roots.push_back(0.0f);
		return roots;
	}

	if (q == 0.0f) {
		// solve bi-quadratic 
		return solveBiQuadratic(1.0f, p, r);
	}

	float p2 = p * p;
	float p3 = p2 * p;
	float p4 = p3 * p;

	float q2 = q * q;
	float q4 = q2 * q2;

	float r2 = r * r;
	float r3 = r2 * r;

	// discriminant
	float h = 256 * r3 - 128 * p2 * r2 + 144 * p * q2 * r - 27 * q4 + 16 * p4 * r - 4 * p3 * q2;

	float a0 = 12 * r + p2;
	float a1 = 4 * r - p2;

	if (h > 0) {
		if (p < 0 && a1 < 0) {
			// 4 real roots
			/*std::vector<float> temp = solveCubic(q2 - 4 * r * p, 8 * r, 4 * p, -8);*/
			std::vector<float> temp = solveCubic(-8, 4 * p, 8 * r, q2 - 4 * r * p);
			float z = temp[0];
			float alpha2 = 2 * z - p;
			float alpha = glm::sqrt(alpha2);
			float signQ = q > 0 ? 1.0f : -1.0f;

			float arg = z * z - r;
			float beta = signQ * glm::sqrt(std::max(arg, 0.0f));
			float D0 = alpha2 - 4 * (z + beta);
			float sqrtD0 = glm::sqrt(std::max(D0, 0.0f));
			float D1 = alpha2 - 4 * (z - beta);
			float sqrtD1 = glm::sqrt(std::max(D1, 0.0f));
			float t0 = 0.5f * (alpha - sqrtD0);
			float t1 = 0.5f * (alpha + sqrtD0);
			float t2 = 0.5f * (-alpha - sqrtD1);
			float t3 = 0.5f * (-alpha + sqrtD1);

			roots.push_back(t0);
			roots.push_back(t1);
			roots.push_back(t2);
			roots.push_back(t3);
		}
		else {
			// 2 complex conjugate pairs

		}
	}
	else if (h < 0) {
		// 2 real roots and 1 complex conjugate pair
		//std::vector<float> temp = solveCubic(q2 - 4 * r * p, 8 * r, 4 * p, -8);
		std::vector<float> temp = solveCubic(-8, 4 * p, 8 * r, q2 - 4 * r * p);
		
		float z = temp[0];
		float alpha2 = 2 * z - p;
		float alpha = glm::sqrt(std::max(alpha2, 0.0f));
		float signQ = q > 0 ? 1.0f : -1.0f;

		float arg = z * z - r;
		float beta = signQ * glm::sqrt(std::max(arg, 0.0f));

		if (signQ > 0.0f) {
			float D1 = alpha2 - 4 * (z - beta);
			float sqrtD1 = glm::sqrt(std::max(D1, 0.0f));

			float t0 = 0.5f * (-alpha - sqrtD1);
			float t1 = 0.5f * (-alpha + sqrtD1);

			roots.push_back(t0);
			roots.push_back(t1);
		}
		else {
			float D0 = alpha2 - 4 * (z + beta);
			float sqrtD0 = glm::sqrt(std::max(D0, 0.0f));

			float t0 = 0.5f * (alpha - sqrtD0);
			float t1 = 0.5f * (alpha + sqrtD0);

			roots.push_back(t0);
			roots.push_back(t1);
		}
	}
	else {
		if (a1 > 0 || (p > 0 && (a1 != 0 || q != 0))) {
			// 1 double real root and 1 complex conjugate pair
			float t0 = -q * a0 / (9 * q2 - 2 * p * a1);

			roots.push_back(t0);
			roots.push_back(t0);
		}
		else {
			if (a0 != 0) {
				// 1 double real root and 2 real roots
				float t0 = -q * a0 / (9 * q2 - 2 * p * a1);
				float alpha = 2 * t0;
				float beta = p + 3 * t0 * t0;
				float discr = alpha * alpha - 4 * beta;
				float temp1 = glm::sqrt(discr);
				float t1 = 0.5f * (-alpha - temp1);
				float t2 = 0.5f * (-alpha + temp1);

				roots.push_back(t0);
				roots.push_back(t0);
				roots.push_back(t1);
				roots.push_back(t2);
			}
			else {
				// 1 triple real root and 1 real roots
				float t0 = -3 * q / (4 * p);
				float t1 = -3 * t0;

				roots.push_back(t0);
				roots.push_back(t0);
				roots.push_back(t0);
				roots.push_back(t1);
			}
		}
	}

	return roots;
}

// Given the linear a1 * x + a0 = 0 
// The solution can be directly found x = -a0 / a1
std::vector<float> PolynomialRoots::solveLinear(float a1, float a0)
{
	std::vector<float> roots;

	if (a1 != 0.0f) {
		roots.push_back(-a0 / a1);
	}

	return roots;
}

// Given the quadratic a2 * x^2 + a1 * x + a0 = 0
// We use the substitution x = t - a1 / (2 * a2) giving the depressed quadratic:
//
// t^2 + p = 0 where,
//
// p = (4*a2*a0 - a1^2) / (4*a2^2)
//
// Define the discriminant of the depressed cubic: 
//
// h = p;
//
// If h > 0 => 0 real root (and 2 complex roots)
// If h < 0 => 2 real roots
// If h == 0 => 1 doube real root of zero
//
// see https://en.wikipedia.org/wiki/Quadratic_equation
std::vector<float> PolynomialRoots::solveQuadratic(float a2, float a1, float a0)
{
	if (a2 == 0.0f) {
		return solveLinear(a1, a0);
	}

	float a0da2 = a0 / a2;
	float a1da2 = a1 / a2;
	float p = a0da2 - 0.25f * a1da2 * a1da2;

	std::vector<float> roots = solveDepressedQuadratic(p);

	for (size_t i = 0; i < roots.size(); i++) {
		roots[i] = roots[i] - a1 / (2 * a2);
	}

	return roots;
}


// A polynomial of the form a4 * x^4 + a2 * x^2 + a0 = 0 is 
// called bi-quadratic. We can solve this by first making the
// substitution z = x^2:
//
// a4 * z^2 + a2 * z + a0 = 0
//
// The if the roots of this quadratic are z0 and z1, then the 4 roots 
// of the bi-quadratic are: sqrt(z0), -sqrt(z0), sqrt(z1), -sqrt(z1).
//
// see https://en.wikipedia.org/wiki/Quartic_function
std::vector<float> PolynomialRoots::solveBiQuadratic(float a4, float a2, float a0)
{
	if (a4 == 0.0f) {
		return solveLinear(a2, a0);
	}

	std::vector<float> roots = solveQuadratic(a4, a2, a0);

	if (roots.size() == 1) {
		float z0 = roots[0];

		roots.clear();
		if (z0 >= 0.0f) {
			float temp = glm::sqrt(z0);

			roots.push_back(temp);
			roots.push_back(-temp);
		}
	}
	else if (roots.size() == 2) {
		float z0 = roots[0];
		float z1 = roots[1];

		roots.clear();
		if (z0 >= 0.0f) {
			float temp = glm::sqrt(z0);

			roots.push_back(temp);
			roots.push_back(-temp);
		}

		if (z1 >= 0.0f) {
			float temp = glm::sqrt(z1);

			roots.push_back(temp);
			roots.push_back(-temp);
		}
	}

	return roots;
}

// Given the cubic a3 * x^3 + a2 * x^2 + a1 * x + a0 = 0
// We use the substitution x = t - a2 / (3 * a3) giving the depressed cubic:
//
// t^3 + p * t + q = 0 where,
//
// p = (3*a3*a1 - a2^2) / (3*a3^2)
// q = (2*a2^3 - 9*a3*a2*a1 + 27*a3^2*a0) / (27*a3^3)
//
// Define the discriminant of the depressed cubic: 
//
// h =  27 * q^2 + 4 * p^3;
//
// If h > 0 => 1 real root (and 2 complex roots)
// If h < 0 => 3 real distinct roots
// If h == 0 => 1 real root and 1 double real root
// see https://en.wikipedia.org/wiki/Cubic_equation
std::vector<float> PolynomialRoots::solveCubic(float a3, float a2, float a1, float a0)
{
	if (a3 == 0.0f) {
		return solveQuadratic(a2, a1, a0);
	}

	float p = (3 * a3 * a1 - a2 * a2) / (3 * a3 * a3);
	float q = (2 * a2 * a2 * a2 - 9 * a3 * a2 * a1 + 27 * a3 * a3 * a0) / (27 * a3 * a3 * a3);

	std::vector<float> roots = solveDepressedCubic(p, q);

	for (size_t i = 0; i < roots.size(); i++) {
		roots[i] = roots[i] - a2 / (3 * a3);
	}

	return roots;
}

// Given the quartic a4 * x^4 + a3 * x^3 + a2 * x^2 + a1 * x + a0 = 0
// We use the substitution x = t - a3 / (4 * a4) giving the depressed quartic:
//
// t^4 + p*t^2 + q * t + r = 0 where,
// 
// p = (8*a2*a4 - 3*a3^2) / (8*a4^2)
// q = (a3^3 - 4*a2*a3*a4 + 8*a1*a4^2) / (8*a4^3)
// r = (-3*a3^4 + 256*a0*a4^3 - 64*a1*a3*a4^2 + 16*a2*a3^2*a4) / (256*a4^4)
//
// Define the discriminant of the depressed quartic: 
//
// h =  256*r^3 - 128*p^2*r^2 + 144*p*q^2*r - 27*q^4 + 16*p^4*r - 4*p^3*q^2
//
// Define the derivatives:
// c(t)    = t ^ 4 + p * t ^ 2 + q * t + r
// c'(t)   = 4*t^3 + 2*p*t + q
// c''(t)  = 12*t^2 + 2*p
// c'''(t) = 24*t
//
//
// see https://en.wikipedia.org/wiki/Quartic_function
std::vector<float> PolynomialRoots::solveQuartic(float a4, float a3, float a2, float a1, float a0)
{
	if (a4 == 0.0f) {
		return solveCubic(a3, a2, a1, a0);
	}

	float a3p2 = a3 * a3;
	float a3p3 = a3p2 * a3;
	float a3p4 = a3p3 * a3;

	float a4p2 = a4 * a4;
	float a4p3 = a4p2 * a4;
	float a4p4 = a4p3 * a4;

	float p = (8 * a2 * a4 - 3 * a3p2) / (8 * a4p4);
	float q = (a3p3 - 4 * a2 * a3 * a4 + 8 * a1 * a4p2) / (8 * a4p3);
	float r = (-3 * a3p4 + 256 * a0 * a4p3 - 64 * a1 * a3 * a4p2 + 16 * a2 * a3p2 * a4) / (256 * a4p4);

	std::vector<float> roots = solveDepressedQuartic(p, q, r);

	for (size_t i = 0; i < roots.size(); i++) {
		roots[i] = roots[i] - a3 / (4 * a4);
	}

	return roots;
}