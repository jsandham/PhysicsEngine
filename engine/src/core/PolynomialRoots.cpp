#include "../../include/core/PolynomialRoots.h"

#include "../../include/glm/glm.hpp"
#include "../../include/glm/detail/func_trigonometric.hpp"
#include "../../include/glm/gtc/constants.hpp"

using namespace PhysicsEngine;

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
// Solve using quadratic formula:
//
// x0 = (-a1 + sqrt(s1^2 - 4 * a2 * a0)) / (2 * a2)
// x1 = (-a1 - sqrt(s1^2 - 4 * a2 * a0)) / (2 * a2)
std::vector<float> PolynomialRoots::solveQuadratic(float a2, float a1, float a0)
{
	if (a2 == 0.0f) {
		return solveLinear(a1, a0);
	}

	std::vector<float> roots;

	float d = a1 * a1 - 4.0f * a2 * a0;
	if (d >= 0.0f) {
		float sqrtd = glm::sqrt(d);

		roots.push_back(0.5f * (-a1 + sqrtd) / a2);
		roots.push_back(0.5f * (-a1 - sqrtd) / a2);
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

	std::vector<float> roots;

	float p = (3 * a3 * a1 - a2 * a2) / (3 * a3 * a3);
	float q = (2 * a2 * a2 * a2 - 9 * a3 * a2 * a1 + 27 * a3 * a3 * a0) / (27 * a3 * a3 * a3);

	const float oneThird = 1.0f / 3.0f;

	if (q == 0) {
		roots.push_back(-a2 / (3 * a3));

		if (p <= 0.0f) {
			float t1 = glm::sqrt(p);
			float t2 = -t1;

			roots.push_back(t1 - a2 / (3 * a3));
			roots.push_back(t2 - a2 / (3 * a3));
		}

		return roots;
	}

	if (p == 0) {
		float t0 = 0;
		if (q > 0.0f) {
			t0 = glm::pow(-q, oneThird);
		}
		else {
			t0 = glm::pow(q, oneThird);
		}

		float x0 = t0 - a2 / (3 * a3);

		roots.push_back(x0);
		return roots;
	}

	// Discriminant
	float h = 27 * q * q + 4 * p * p * p;

	if (h > 0.0f) {
		// One real root
		float sqrth = glm::sqrt(h / 108);
		float R = -0.5f * q + sqrth;
		float T = -0.5f * q - sqrth;

		float S = glm::pow(std::abs(R), oneThird);
		if (R < 0.0f) {
			S *= -1.0f;
		}

		float Q = glm::pow(glm::abs(T), oneThird);
		if (T < 0.0f) {
			Q *= -1.0f;
		}

		float x0 = S + Q - a2 / (3 * a3);

		roots.push_back(x0);
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

		float temp = a2 / (3 * a3);

		float x0 = t0 - temp;
		float x1 = t1 - temp;
		float x2 = t2 - temp;

		roots.push_back(x0);
		roots.push_back(x1);
		roots.push_back(x2);
		return roots;
	}
	else {
		// One real root and one real double root
		float t0 = 3 * q / p;
		float t1 = -1.5f * q / p;
		float t2 = t1;

		float temp = a2 / (3 * a3);

		float x0 = t0 - temp;
		float x1 = t1 - temp;
		float x2 = t2 - temp;

		roots.push_back(x0);
		roots.push_back(x1);
		roots.push_back(x2);
		return roots;
	}
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
// see https://en.wikipedia.org/wiki/Quartic_function
std::vector<float> PolynomialRoots::solveQuartic(float a4, float a3, float a2, float a1, float a0)
{
	if (a4 == 0.0f) {
		return solveCubic(a3, a2, a1, a0);
	}

	std::vector<float> roots;

	float a3p2 = a3 * a3;
	float a3p3 = a3p2 * a3;
	float a3p4 = a3p3 * a3;

	float a4p2 = a4 * a4;
	float a4p3 = a4p2 * a4;
	float a4p4 = a4p3 * a4;

	float p = (8 * a2 * a4 - 3 * a3p2) / (8 * a4p4);
	float q = (a3p3 - 4 * a2 * a3 * a4 + 8 * a1 * a4p2) / (8 * a4p3);
	float r = (-3 * a3p4 + 256 * a0 * a4p3 - 64 * a1 * a3 * a4p2 + 16 * a2 * a3p2 * a4) / (256 * a4p4);

	if (r == 0.0f) {
		// solve depressed cubic
		roots = solveCubic(1.0f, 0.0f, p, q);

		// transform roots from depressed cubic
		for (size_t i = 0; i < roots.size(); i++) {
			roots[i] = roots[i] - a3 / (4 * a4);
		}

		// add "t==0" root
		roots.push_back(-a3 / (4 * a4));
		return roots;
	}

	if (q == 0) {

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


	return roots;
}