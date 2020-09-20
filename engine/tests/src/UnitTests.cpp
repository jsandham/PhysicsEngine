#include <iostream>
#include <vector>

#include "../include/UnitTests.h"

#include <glm/glm.hpp>

#include <core/Log.h>
#include <core/PoolAllocator.h>
#include <core/Intersect.h>
#include <core/ClosestDistance.h>
#include <core/PolynomialRoots.h>
#include <core/Physics.h>

#include <components/Transform.h>

using namespace Testing;
using namespace PhysicsEngine;

void UnitTests::print(const std::string testMessage, bool testPassed)
{
	if (testPassed) {
		std::string message = "CORE UNIT TEST: " + testMessage + " PASSED\n";
		std::cout << message;
		//Log::info(message.c_str());
	}
	else {
		std::string message = "CORE UNIT TEST: " + testMessage + " FAILED\n";
		std::cout << message;
		//Log::error(message.c_str());
	}
}

void UnitTests::run()
{
	poolAllocatorTest0();
	poolAllocatorTest1();
	poolAllocatorTest2();
	poolAllocatorTest3();
	poolAllocatorTest4();

	polynomialRootsTest0();

	rayPlaneIntersectionTest0();
	rayPlaneIntersectionTest1();
	rayPlaneIntersectionTest2();
	rayPlaneIntersectionTest3();
	rayPlaneIntersectionTest4();
	rayPlaneIntersectionTest5();
	rayPlaneIntersectionTest6();
	rayPlaneIntersectionTest7();
	rayPlaneIntersectionTest8();
	rayPlaneIntersectionTest9();

	raySphereIntersectionTest0();
	raySphereIntersectionTest1();
	raySphereIntersectionTest2();
	raySphereIntersectionTest3();
	raySphereIntersectionTest4();
	raySphereIntersectionTest5();
	raySphereIntersectionTest6();
	raySphereIntersectionTest7();
	raySphereIntersectionTest8();
	raySphereIntersectionTest9();
	raySphereIntersectionTest10();
	raySphereIntersectionTest11();
	raySphereIntersectionTest12();
	raySphereIntersectionTest13();
	raySphereIntersectionTest14();
	raySphereIntersectionTest15();
	raySphereIntersectionTest16();
	raySphereIntersectionTest17();
	raySphereIntersectionTest18();
	raySphereIntersectionTest19();
	raySphereIntersectionTest20();
	raySphereIntersectionTest21();
	raySphereIntersectionTest22();

	rayAABBIntersectionTest0();
	rayAABBIntersectionTest1();
	rayAABBIntersectionTest2();
	rayAABBIntersectionTest3();
	rayAABBIntersectionTest4();
	rayAABBIntersectionTest5();
	rayAABBIntersectionTest6();
	rayAABBIntersectionTest7();
	rayAABBIntersectionTest8();
	rayAABBIntersectionTest9();
	rayAABBIntersectionTest10();
	rayAABBIntersectionTest11();
	rayAABBIntersectionTest12();
	rayAABBIntersectionTest13();
	rayAABBIntersectionTest14();
	rayAABBIntersectionTest15();
	rayAABBIntersectionTest16();
	rayAABBIntersectionTest17();
	rayAABBIntersectionTest18();

	sphereAABBIntersectionTest0();
	sphereAABBIntersectionTest1();
	sphereAABBIntersectionTest2();
	sphereAABBIntersectionTest3();
	sphereAABBIntersectionTest4();
	sphereAABBIntersectionTest5();
	sphereAABBIntersectionTest6();
	sphereAABBIntersectionTest7();
	sphereAABBIntersectionTest8();
	sphereAABBIntersectionTest9();
	sphereAABBIntersectionTest10();
	sphereAABBIntersectionTest11();
	sphereAABBIntersectionTest12();
	sphereAABBIntersectionTest13();
	sphereAABBIntersectionTest14();
	sphereAABBIntersectionTest15();
	sphereAABBIntersectionTest16();
	sphereAABBIntersectionTest17();
	sphereAABBIntersectionTest18();

	/*sphereSphereIntersectionTest0();
	sphereSphereIntersectionTest1();
	sphereSphereIntersectionTest2();
	sphereSphereIntersectionTest3();
	sphereSphereIntersectionTest4();
	sphereSphereIntersectionTest5();
	sphereSphereIntersectionTest6();
	sphereSphereIntersectionTest7();
	sphereSphereIntersectionTest8();
	sphereSphereIntersectionTest9();
	sphereSphereIntersectionTest10();
	sphereSphereIntersectionTest11();
	sphereSphereIntersectionTest12();*/

	sphereFrustumIntersectionTest0();
	sphereFrustumIntersectionTest1();
	sphereFrustumIntersectionTest2();
	sphereFrustumIntersectionTest3();
	sphereFrustumIntersectionTest4();
	sphereFrustumIntersectionTest5();
	sphereFrustumIntersectionTest6();
	sphereFrustumIntersectionTest7();
	sphereFrustumIntersectionTest8();
	sphereFrustumIntersectionTest9();
	sphereFrustumIntersectionTest10();
	sphereFrustumIntersectionTest11();
	sphereFrustumIntersectionTest12();
	sphereFrustumIntersectionTest13();

	aabbAABBIntersectionTest0();
	aabbAABBIntersectionTest1();
	aabbAABBIntersectionTest2();
	aabbAABBIntersectionTest3();
	aabbAABBIntersectionTest4();
	aabbAABBIntersectionTest5();
	aabbAABBIntersectionTest6();
	aabbAABBIntersectionTest7();
	aabbAABBIntersectionTest8();
	aabbAABBIntersectionTest9();
	aabbAABBIntersectionTest10();
}

void UnitTests::poolAllocatorTest0()
{
	PoolAllocator<Transform> allocator;

	for (int i = 0; i < 50; i++) {
		allocator.construct();
	}

	for (int i = 0; i < 50; i++) {
		allocator.destruct(0);
	}

	print("Pool allocator test (0)", allocator.getCount() == 0);
}

void UnitTests::poolAllocatorTest1()
{
	PoolAllocator<Transform> allocator;

	for (int i = 0; i < 50; i++) {
		allocator.construct();
	}

	for (int i = 0; i < 50; i++) {
		allocator.destruct(allocator.getCount() - 1);
	}

	print("Pool allocator test (1)", allocator.getCount() == 0);
}

void UnitTests::poolAllocatorTest2()
{
	PoolAllocator<Transform> allocator;

	for (int i = 0; i < 500; i++) {
		allocator.construct();
	}

	for (int i = 0; i < 500; i++) {
		allocator.destruct(allocator.getCount() - 1);
	}

	print("Pool allocator test (2)", allocator.getCount() == 0);
}

void UnitTests::poolAllocatorTest3()
{
	PoolAllocator<Transform> allocator;

	for (int i = 0; i < 2000; i++) {
		allocator.construct();
		allocator.destruct(0);
	}

	print("Pool allocator test (3)", allocator.getCount() == 0);
}

void UnitTests::poolAllocatorTest4()
{
	PoolAllocator<Transform> allocator;

	for (int i = 0; i < 2000; i++) {
		allocator.construct();
	}

	while (allocator.getCount() > 0) {
		int indexToDelete = rand() % allocator.getCount();

		allocator.destruct(indexToDelete);
	}

	print("Pool allocator test (4)", allocator.getCount() == 0);
}

void UnitTests::poolAllocatorTest5()
{
	PoolAllocator<Transform> allocator;

	for (int i = 0; i < 2000; i++) {
		allocator.construct();
	}

	while (allocator.getCount() > 1000) {
		int indexToDelete = rand() % allocator.getCount();

		allocator.destruct(indexToDelete);
	}

	for (int i = 0; i < 1000; i++) {
		allocator.construct();
	}

	while (allocator.getCount() > 0) {
		int indexToDelete = rand() % allocator.getCount();

		allocator.destruct(indexToDelete);
	}


	print("Pool allocator test (4)", allocator.getCount() == 0);
}


void UnitTests::polynomialRootsTest0()
{
	//std::vector<float> roots = PolynomialRoots::solveDepressedQuartic(-6.9, -0.9, 0.57);
	//std::vector<float> roots = PolynomialRoots::solveQuartic(0, 0, 20, -1, -1);

	std::vector<float> roots = PolynomialRoots::solveDepressedQuadratic(-0.050625);
	//std::vector<float> roots = PolynomialRoots::solveQuadratic(20, -1, -1);

	for (size_t i = 0; i < roots.size(); i++) {
		std::cout << roots[i] << std::endl;
	}
}





void UnitTests::rayPlaneIntersectionTest0()
{
	Ray ray;
	ray.mOrigin = glm::vec3(0, 0, 0);
	ray.mDirection = glm::vec3(-1, 0, 0);

	Plane plane;
	plane.mNormal = glm::vec3(1, 0, 0);
	plane.mX0 = glm::vec3(2, 0, 0);

	bool answer = Intersect::intersect(ray, plane);

	print("Intersection of ray and plane (0) ", (answer == false));
}

void UnitTests::rayPlaneIntersectionTest1()
{
	Ray ray;
	ray.mOrigin = glm::vec3(4, 0, 0);
	ray.mDirection = glm::vec3(-1, 0, 0);

	Plane plane;
	plane.mNormal = glm::vec3(1, 0, 0);
	plane.mX0 = glm::vec3(2, 0, 0);

	bool answer = Intersect::intersect(ray, plane);

	print("Intersection of ray and plane (1) ", (answer == true));
}

void UnitTests::rayPlaneIntersectionTest2()
{
	Ray ray;
	ray.mOrigin = glm::vec3(0, 0, 0);
	ray.mDirection = glm::vec3(1, 1, 1);

	Plane plane;
	plane.mNormal = glm::vec3(-1, -1, -1);
	plane.mX0 = glm::vec3(-0.1, -0.1, -0.1);

	bool answer = Intersect::intersect(ray, plane);

	print("Intersection of ray and plane (2) ", (answer == false));
}

void UnitTests::rayPlaneIntersectionTest3()
{
	Ray ray;
	ray.mOrigin = glm::vec3(0, 0, 0);
	ray.mDirection = glm::vec3(1, 1, 1);

	Plane plane;
	plane.mNormal = glm::vec3(-1, -1, -1);
	plane.mX0 = glm::vec3(1.1, 1.1, 1.1);

	bool answer = Intersect::intersect(ray, plane);

	print("Intersection of ray and plane (3) ", (answer == true));
}

void UnitTests::rayPlaneIntersectionTest4()
{
	Ray ray;
	ray.mOrigin = glm::vec3(1, 1, 1);
	ray.mDirection = glm::vec3(1, 0, 0);

	Plane plane;
	plane.mNormal = glm::vec3(0, 0, 1);
	plane.mX0 = glm::vec3(0, 0, 1.1);

	bool answer = Intersect::intersect(ray, plane);

	print("Intersection of ray and plane (4) ", (answer == false));
}

void UnitTests::rayPlaneIntersectionTest5()
{
	Ray ray;
	ray.mOrigin = glm::vec3(0, 0, 0);
	ray.mDirection = glm::vec3(1, 1, 1);

	Plane plane;
	plane.mNormal = glm::vec3(0.2, 0.7, 1.3);
	plane.mX0 = glm::vec3(1, 1, 1);

	bool answer = Intersect::intersect(ray, plane);

	print("Intersection of ray and plane (5) ", (answer == true));
}

void UnitTests::rayPlaneIntersectionTest6()
{
	Ray ray;
	ray.mOrigin = glm::vec3(0, 0, 0);
	ray.mDirection = glm::vec3(1, 1, 1);

	Plane plane;
	plane.mNormal = glm::vec3(-0.4, -0.7, 0.9);
	plane.mX0 = glm::vec3(1, 1, 1);

	bool answer = Intersect::intersect(ray, plane);

	print("Intersection of ray and plane (6) ", (answer == true));
}

void UnitTests::rayPlaneIntersectionTest7()
{
	Ray ray;
	ray.mOrigin = glm::vec3(0, 0, 0);
	ray.mDirection = glm::vec3(1, 1, 1);

	Plane plane;
	plane.mNormal = glm::vec3(0.7, -0.6, -0.3);
	plane.mX0 = glm::vec3(1, 1, 1);

	bool answer = Intersect::intersect(ray, plane);

	print("Intersection of ray and plane (7) ", (answer == true));
}

void UnitTests::rayPlaneIntersectionTest8()
{
	Ray ray;
	ray.mOrigin = glm::vec3(0, 0, 0);
	ray.mDirection = glm::vec3(1, 1, 1);

	Plane plane;
	plane.mNormal = glm::vec3(0.7, 0.6, 0.3);
	plane.mX0 = glm::vec3(-1, -1, -1);

	bool answer = Intersect::intersect(ray, plane);

	print("Intersection of ray and plane (8) ", (answer == false));
}

void UnitTests::rayPlaneIntersectionTest9()
{
	Ray ray;
	ray.mOrigin = glm::vec3(0, 0, 0);
	ray.mDirection = glm::vec3(1, 1, 1);

	Plane plane;
	plane.mNormal = glm::vec3(0.5, 0.9, 0.1);
	plane.mX0 = glm::vec3(-1, -1, -1);

	bool answer = Intersect::intersect(ray, plane);

	print("Intersection of ray and plane (9) ", (answer == false));
}

void UnitTests::raySphereIntersectionTest0()
{
	Ray ray(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(1.0f, 0.0f, 0.0f));
	Sphere sphere(glm::vec3(10.0f, 0.0f, 0.0f), 1.0f);

	bool answer = Intersect::intersect(ray, sphere);

	print("Intersection of ray and sphere (0) ", (answer == true));
}

void UnitTests::raySphereIntersectionTest1()
{
	Ray ray(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
	Sphere sphere(glm::vec3(0.0f, 10.0f, 0.0f), 1.0f);

	bool answer = Intersect::intersect(ray, sphere);

	print("Intersection of ray and sphere (1) ", (answer == true));
}

void UnitTests::raySphereIntersectionTest2()
{
	Ray ray(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f));
	Sphere sphere(glm::vec3(0.0f, 0.0f, 10.0f), 1.0f);

	bool answer = Intersect::intersect(ray, sphere);

	print("Intersection of ray and sphere (2) ", (answer == true));
}

void UnitTests::raySphereIntersectionTest3()
{
	Ray ray(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(-1.0f, 0.0f, 0.0f));
	Sphere sphere(glm::vec3(-10.0f, 0.0f, 0.0f), 1.0f);

	bool answer = Intersect::intersect(ray, sphere);

	print("Intersection of ray and sphere (3) ", (answer == true));
}

void UnitTests::raySphereIntersectionTest4()
{
	Ray ray(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, -1.0f, 0.0f));
	Sphere sphere(glm::vec3(0.0f, -10.0f, 0.0f), 1.0f);

	bool answer = Intersect::intersect(ray, sphere);

	print("Intersection of ray and sphere (4) ", (answer == true));
}

void UnitTests::raySphereIntersectionTest5()
{
	Ray ray(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, -1.0f));
	Sphere sphere(glm::vec3(0.0f, 0.0f, -10.0f), 1.0f);

	bool answer = Intersect::intersect(ray, sphere);

	print("Intersection of ray and sphere (5) ", (answer == true));
}

void UnitTests::raySphereIntersectionTest6()
{
	Ray ray(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(1.0f, 1.0f, 1.0f));
	Sphere sphere(glm::vec3(10.0f, 10.0f, 10.0f), 1.0f);

	bool answer = Intersect::intersect(ray, sphere);

	print("Intersection of ray and sphere (6) ", (answer == true));
}

void UnitTests::raySphereIntersectionTest7()
{
	Ray ray(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(-1.0f, -1.0f, -1.0f));
	Sphere sphere(glm::vec3(-10.0f, -10.0f, -10.0f), 1.0f);

	bool answer = Intersect::intersect(ray, sphere);

	print("Intersection of ray and sphere (7) ", (answer == true));
}

void UnitTests::raySphereIntersectionTest8()
{
	Ray ray(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(10.0f, 0.5f, 0.5f));
	Sphere sphere(glm::vec3(10.0f, 0.0f, 0.0f), 1.0f);

	bool answer = Intersect::intersect(ray, sphere);

	print("Intersection of ray and sphere (8) ", (answer == true));
}

void UnitTests::raySphereIntersectionTest9()
{
	Ray ray(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(10.0f, 1.0f, 0.0f));
	Sphere sphere(glm::vec3(10.0f, 0.0f, 0.0f), 1.0f);

	bool answer = Intersect::intersect(ray, sphere);

	print("Intersection of ray and sphere (9) ", (answer == true));
}

void UnitTests::raySphereIntersectionTest10()
{
	Ray ray(glm::vec3(1.0f, 1.0f, 1.0f), glm::vec3(1.0f, 0.0f, 0.0f));
	Sphere sphere(glm::vec3(10.0f, 0.0f, 0.0f), 2.0f);

	bool answer = Intersect::intersect(ray, sphere);

	print("Intersection of ray and sphere (10) ", (answer == true));
}

void UnitTests::raySphereIntersectionTest11()
{
	Ray ray(glm::vec3(-1.0f, -1.0f, -1.0f), glm::vec3(1.0f, 0.0f, 0.0f));
	Sphere sphere(glm::vec3(10.0f, 0.0f, 0.0f), 2.0f);

	bool answer = Intersect::intersect(ray, sphere);

	print("Intersection of ray and sphere (11) ", (answer == true));
}

void UnitTests::raySphereIntersectionTest12()
{
	Ray ray(glm::vec3(1.0f, 2.0f, 3.0f), glm::vec3(-1.0f, -2.0f, -3.0f));
	Sphere sphere(glm::vec3(0.0f, 0.0f, 0.0f), 1.0f);

	bool answer = Intersect::intersect(ray, sphere);

	print("Intersection of ray and sphere (12) ", (answer == true));
}

void UnitTests::raySphereIntersectionTest13()
{
	Ray ray(glm::vec3(-1.0f, -2.0f, -3.0f), glm::vec3(1.0f, 2.0f, 3.0f));
	Sphere sphere(glm::vec3(0.0f, 0.0f, 0.0f), 1.0f);

	bool answer = Intersect::intersect(ray, sphere);

	print("Intersection of ray and sphere (13) ", (answer == true));
}

void UnitTests::raySphereIntersectionTest14()
{
	Ray ray(glm::vec3(1.0f, 2.0f, 3.0f), glm::vec3(-1.0f, -2.0f, -3.0f));
	Sphere sphere(glm::vec3(0.2f, 0.3f, 0.1f), 2.0f);

	bool answer = Intersect::intersect(ray, sphere);

	print("Intersection of ray and sphere (14) ", (answer == true));
}

void UnitTests::raySphereIntersectionTest15()
{
	Ray ray(glm::vec3(-1.0f, -2.0f, -3.0f), glm::vec3(1.0f, 2.0f, 3.0f));
	Sphere sphere(glm::vec3(0.2f, 0.3f, 0.1f), 2.0f);

	bool answer = Intersect::intersect(ray, sphere);

	print("Intersection of ray and sphere (15) ", (answer == true));
}

void UnitTests::raySphereIntersectionTest16()
{
	Ray ray(glm::vec3(-1.0f, 2.0f, -3.0f), glm::vec3(4.0f, -2.0f, 3.0f));
	Sphere sphere(glm::vec3(-1.0f, 2.0f, -3.0f), 0.5f);

	bool answer = Intersect::intersect(ray, sphere);

	print("Intersection of ray and sphere (16) ", (answer == true));
}

void UnitTests::raySphereIntersectionTest17()
{
	Ray ray(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(1.0f, 0.0f, 0.0f));
	Sphere sphere(glm::vec3(-10.0f, 0.0f, 0.0f), 1.0f);

	bool answer = Intersect::intersect(ray, sphere);

	print("Intersection of ray and sphere (17) ", (answer == false));
}

void UnitTests::raySphereIntersectionTest18()
{
	Ray ray(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
	Sphere sphere(glm::vec3(0.0f, -10.0f, 0.0f), 1.0f);

	bool answer = Intersect::intersect(ray, sphere);

	print("Intersection of ray and sphere (18) ", (answer == false));
}

void UnitTests::raySphereIntersectionTest19()
{
	Ray ray(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f));
	Sphere sphere(glm::vec3(0.0f, 0.0f, -10.0f), 1.0f);

	bool answer = Intersect::intersect(ray, sphere);

	print("Intersection of ray and sphere (19) ", (answer == false));
}

void UnitTests::raySphereIntersectionTest20()
{
	Ray ray(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(1.0f, 1.0f, 1.0f));
	Sphere sphere(glm::vec3(-10.0f, -10.0f, -10.0f), 1.0f);

	bool answer = Intersect::intersect(ray, sphere);

	print("Intersection of ray and sphere (20) ", (answer == false));
}

void UnitTests::raySphereIntersectionTest21()
{
	Ray ray(glm::vec3(2.0f, 0.0f, 0.0f), glm::vec3(1.0f, 0.0f, 0.0f));
	Sphere sphere(glm::vec3(1.0f, 0.0f, 0.0f), 0.5f);

	bool answer = Intersect::intersect(ray, sphere);

	print("Intersection of ray and sphere (21) ", (answer == false));
}

void UnitTests::raySphereIntersectionTest22()
{
	Ray ray(glm::vec3(1.0f, 1.0f, 1.0f), glm::vec3(1.0f, 0.0f, 0.0f));
	Sphere sphere(glm::vec3(10.0f, 0.0f, 0.0f), 1.0f);

	bool answer = Intersect::intersect(ray, sphere);

	print("Intersection of ray and sphere (22) ", (answer == false));
}

void UnitTests::rayAABBIntersectionTest0()
{
	Ray ray(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(1.0f, 0.0f, 0.0f));
	AABB aabb(glm::vec3(10.0f, 0.0f, 0.0f), glm::vec3(1.0f, 1.0f, 1.0f));

	bool answer = Intersect::intersect(ray, aabb);

	print("Intersection of ray and AABB (0) ", (answer == true));
}

void UnitTests::rayAABBIntersectionTest1()
{
	Ray ray(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
	AABB aabb(glm::vec3(0.0f, 10.0f, 0.0f), glm::vec3(1.0f, 1.0f, 1.0f));

	bool answer = Intersect::intersect(ray, aabb);

	print("Intersection of ray and AABB (1) ", (answer == true));
}

void UnitTests::rayAABBIntersectionTest2()
{
	Ray ray(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f));
	AABB aabb(glm::vec3(0.0f, 0.0f, 10.0f), glm::vec3(1.0f, 1.0f, 1.0f));

	bool answer = Intersect::intersect(ray, aabb);

	print("Intersection of ray and AABB (2) ", (answer == true));
}

void UnitTests::rayAABBIntersectionTest3()
{
	Ray ray(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(-1.0f, 0.0f, 0.0f));
	AABB aabb(glm::vec3(-10.0f, 0.0f, 0.0f), glm::vec3(1.0f, 1.0f, 1.0f));

	bool answer = Intersect::intersect(ray, aabb);

	print("Intersection of ray and AABB (3) ", (answer == true));
}

void UnitTests::rayAABBIntersectionTest4()
{
	Ray ray(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, -1.0f, 0.0f));
	AABB aabb(glm::vec3(0.0f, -10.0f, 0.0f), glm::vec3(1.0f, 1.0f, 1.0f));

	bool answer = Intersect::intersect(ray, aabb);

	print("Intersection of ray and AABB (4) ", (answer == true));
}

void UnitTests::rayAABBIntersectionTest5()
{
	Ray ray(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, -1.0f));
	AABB aabb(glm::vec3(0.0f, 0.0f, -10.0f), glm::vec3(1.0f, 1.0f, 1.0f));

	bool answer = Intersect::intersect(ray, aabb);

	print("Intersection of ray and AABB (5) ", (answer == true));
}

void UnitTests::rayAABBIntersectionTest6()
{
	Ray ray(glm::vec3(1.0f, 1.0f, 1.0f), glm::vec3(1.0f, 0.0f, 0.0f));
	AABB aabb(glm::vec3(2.0f, 0.0f, 0.0f), glm::vec3(2.001f, 2.0001f, 2.001f));

	bool answer = Intersect::intersect(ray, aabb);

	print("Intersection of ray and AABB (6) ", (answer == true));
}

void UnitTests::rayAABBIntersectionTest7()
{
	Ray ray(glm::vec3(1.0f, 1.0f, 1.0f), glm::vec3(1.0f, 0.0f, 0.0f));
	AABB aabb(glm::vec3(2.0f, 0.0f, 0.0f), glm::vec3(3.0f, 4.0f, 2.00001f));

	bool answer = Intersect::intersect(ray, aabb);

	print("Intersection of ray and AABB (7) ", (answer == true));
}

void UnitTests::rayAABBIntersectionTest8()
{
	Ray ray(glm::vec3(0.0f, 1.0f, 0.0f), glm::vec3(-1.0f, -1.0f, -1.0f));
	AABB aabb(glm::vec3(0.0f, -2.0f, 0.0f), glm::vec3(4.0f, 4.0f, 4.0f));

	bool answer = Intersect::intersect(ray, aabb);

	print("Intersection of ray and AABB (8) ", (answer == true));
}

void UnitTests::rayAABBIntersectionTest9()
{
	Ray ray(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(1.0f, 0.0f, 0.0f));
	AABB aabb(glm::vec3(-1.0f, 0.0f, 0.0f), glm::vec3(4.0f, 4.0f, 4.0f));

	bool answer = Intersect::intersect(ray, aabb);

	print("Intersection of ray and AABB (9) ", (answer == true));
}

void UnitTests::rayAABBIntersectionTest10()
{
	Ray ray(glm::vec3(-2.0f, 0.0f, 0.0f), glm::vec3(1.0f, 0.0f, 0.0f));
	AABB aabb(glm::vec3(-1.0f, 0.0f, 0.0f), glm::vec3(4.0f, 4.0f, 4.0f));

	bool answer = Intersect::intersect(ray, aabb);

	print("Intersection of ray and AABB (10) ", (answer == true));
}

void UnitTests::rayAABBIntersectionTest11()
{
	Ray ray(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(1.0f, 0.0f, 0.0f));
	AABB aabb(glm::vec3(-1.0f, 0.0f, 0.0f), glm::vec3(1.0f, 1.0f, 1.0f));

	bool answer = Intersect::intersect(ray, aabb);

	print("Intersection of ray and AABB (11) ", (answer == false));
}

void UnitTests::rayAABBIntersectionTest12()
{
	Ray ray(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
	AABB aabb(glm::vec3(0.0f, -1.0f, 0.0f), glm::vec3(1.0f, 1.0f, 1.0f));

	bool answer = Intersect::intersect(ray, aabb);

	print("Intersection of ray and AABB (12) ", (answer == false));
}

void UnitTests::rayAABBIntersectionTest13()
{
	Ray ray(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f));
	AABB aabb(glm::vec3(0.0f, 0.0f, -1.0f), glm::vec3(1.0f, 1.0f, 1.0f));

	bool answer = Intersect::intersect(ray, aabb);

	print("Intersection of ray and AABB (13) ", (answer == false));
}

void UnitTests::rayAABBIntersectionTest14()
{
	Ray ray(glm::vec3(1.0f, 1.0f, 1.0f), glm::vec3(0.5f, 0.4f, 1.3f));
	AABB aabb(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(1.0f, 1.0f, 1.0f));

	bool answer = Intersect::intersect(ray, aabb);

	print("Intersection of ray and AABB (14) ", (answer == false));
}

void UnitTests::rayAABBIntersectionTest15()
{
	Ray ray(glm::vec3(1.0f, 1.0f, 1.0f), glm::vec3(-0.5f, -0.4f, -1.3f));
	AABB aabb(glm::vec3(2.0f, 2.0f, 2.0f), glm::vec3(1.0f, 1.0f, 1.0f));

	bool answer = Intersect::intersect(ray, aabb);

	print("Intersection of ray and AABB (15) ", (answer == false));
}

void UnitTests::rayAABBIntersectionTest16()
{
	Ray ray(glm::vec3(1.0f, 1.0f, 1.0f), glm::vec3(1.0f, 0.0f, 0.0f));
	AABB aabb(glm::vec3(2.0f, 0.0f, 0.0f), glm::vec3(2.00001f, 2.00001f, 2.00001f));

	bool answer = Intersect::intersect(ray, aabb);

	print("Intersection of ray and AABB (16) ", (answer == true));
}

void UnitTests::rayAABBIntersectionTest17()
{
	Ray ray(glm::vec3(1.0f, 1.0f, 1.0f), glm::vec3(1.0f, 0.0f, 0.0f));
	AABB aabb(glm::vec3(2.0f, 0.0f, 0.0f), glm::vec3(3.0f, 4.0f, 2.00001f));

	bool answer = Intersect::intersect(ray, aabb);

	print("Intersection of ray and AABB (17) ", (answer == true));
}

void UnitTests::rayAABBIntersectionTest18()
{
	Ray ray(glm::vec3(-1.0f, 10.0f, 2.2f), glm::vec3(1.0f, 0.1f, -2.3f));
	AABB aabb(glm::vec3(-1.015f, 9.85f, 1.9f), glm::vec3(2.8f, 1.234f, 2.1f));

	bool answer = Intersect::intersect(ray, aabb);

	print("Intersection of ray and AABB (18) ", (answer == true));
}


void UnitTests::sphereAABBIntersectionTest0()
{
	Sphere sphere(glm::vec3(0.0f, 0.0f, 0.0f), 1.0f);
	AABB aabb(glm::vec3(1.0f, 0.0f, 0.0f), glm::vec3(1.0f, 1.0f, 1.0f));

	bool answer = Intersect::intersect(sphere, aabb);

	print("Intersection of sphere and AABB (0) ", (answer == true));
}

void UnitTests::sphereAABBIntersectionTest1()
{
	Sphere sphere(glm::vec3(0.0f, 0.0f, 0.0f), 1.0f);
	AABB aabb(glm::vec3(1.0f, 0.0f, 0.0f), glm::vec3(1.0f, 1.0f, 1.0f));

	bool answer = Intersect::intersect(sphere, aabb);

	print("Intersection of sphere and AABB (1) ", (answer == true));
}

void UnitTests::sphereAABBIntersectionTest2()
{
	Sphere sphere(glm::vec3(0.0f, 0.0f, 0.0f), 1.0f);
	AABB aabb(glm::vec3(-1.0f, 0.0f, 0.0f), glm::vec3(1.0f, 1.0f, 1.0f));

	bool answer = Intersect::intersect(sphere, aabb);

	print("Intersection of sphere and AABB (2) ", (answer == true));
}

void UnitTests::sphereAABBIntersectionTest3()
{
	Sphere sphere(glm::vec3(0.0f, 0.0f, 0.0f), 1.0f);
	AABB aabb(glm::vec3(0.0f, 1.0f, 0.0f), glm::vec3(1.0f, 1.0f, 1.0f));

	bool answer = Intersect::intersect(sphere, aabb);

	print("Intersection of sphere and AABB (3) ", (answer == true));
}

void UnitTests::sphereAABBIntersectionTest4()
{
	Sphere sphere(glm::vec3(0.0f, 0.0f, 0.0f), 1.0f);
	AABB aabb(glm::vec3(0.0f, -1.0f, 0.0f), glm::vec3(1.0f, 1.0f, 1.0f));

	bool answer = Intersect::intersect(sphere, aabb);

	print("Intersection of sphere and AABB (4) ", (answer == true));
}

void UnitTests::sphereAABBIntersectionTest5()
{
	Sphere sphere(glm::vec3(0.0f, 0.0f, 0.0f), 1.0f);
	AABB aabb(glm::vec3(0.0f, 0.0f, 1.0f), glm::vec3(1.0f, 1.0f, 1.0f));

	bool answer = Intersect::intersect(sphere, aabb);

	print("Intersection of sphere and AABB (5) ", (answer == true));
}

void UnitTests::sphereAABBIntersectionTest6()
{
	Sphere sphere(glm::vec3(0.0f, 0.0f, 0.0f), 1.0f);
	AABB aabb(glm::vec3(0.0f, 0.0f, -1.0f), glm::vec3(1.0f, 1.0f, 1.0f));

	bool answer = Intersect::intersect(sphere, aabb);

	print("Intersection of sphere and AABB (6) ", (answer == true));
}

void UnitTests::sphereAABBIntersectionTest7()
{
	Sphere sphere(glm::vec3(0.0f, 0.0f, 0.0f), 1.0f);
	AABB aabb(glm::vec3(1.0f, 1.0f, 1.0f), glm::vec3(1.0f, 1.0f, 1.0f));

	bool answer = Intersect::intersect(sphere, aabb);

	print("Intersection of sphere and AABB (7) ", (answer == true));
}

void UnitTests::sphereAABBIntersectionTest8()
{
	Sphere sphere(glm::vec3(0.0f, 0.0f, 0.0f), 1.0f);
	AABB aabb(glm::vec3(-1.0f, -1.0f, -1.0f), glm::vec3(1.0f, 1.0f, 1.0f));

	bool answer = Intersect::intersect(sphere, aabb);

	print("Intersection of sphere and AABB (8) ", (answer == true));
}

void UnitTests::sphereAABBIntersectionTest9()
{
	Sphere sphere(glm::vec3(0.0f, 0.0f, 0.0f), 1.0f);
	AABB aabb(glm::vec3(-1.0f, 1.0f, -1.0f), glm::vec3(1.0f, 1.0f, 1.0f));

	bool answer = Intersect::intersect(sphere, aabb);

	print("Intersection of sphere and AABB (9) ", (answer == true));
}

void UnitTests::sphereAABBIntersectionTest10()
{
	Sphere sphere(glm::vec3(0.0f, 0.0f, 0.0f), 1.0f);
	AABB aabb(glm::vec3(1.0f, -1.0f, 1.0f), glm::vec3(1.0f, 1.0f, 1.0f));

	bool answer = Intersect::intersect(sphere, aabb);

	print("Intersection of sphere and AABB (10) ", (answer == true));
}

void UnitTests::sphereAABBIntersectionTest11()
{
	Sphere sphere(glm::vec3(0.0f, 0.0f, 0.0f), 1.0f);
	AABB aabb(glm::vec3(1.0f, -1.0f, -1.0f), glm::vec3(1.0f, 1.0f, 1.0f));

	bool answer = Intersect::intersect(sphere, aabb);

	print("Intersection of sphere and AABB (11) ", (answer == true));
}

void UnitTests::sphereAABBIntersectionTest12()
{
	Sphere sphere(glm::vec3(0.0f, 0.0f, 0.0f), 1.0f);
	AABB aabb(glm::vec3(-1.0f, -1.0f, 1.0f), glm::vec3(1.0f, 1.0f, 1.0f));

	bool answer = Intersect::intersect(sphere, aabb);

	print("Intersection of sphere and AABB (12) ", (answer == true));
}

void UnitTests::sphereAABBIntersectionTest13()
{
	Sphere sphere(glm::vec3(0.0f, 0.0f, 0.0f), 1.0f);
	AABB aabb(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.2f, 0.2f, 0.2f));

	bool answer = Intersect::intersect(sphere, aabb);

	print("Intersection of sphere and AABB (13) ", (answer == true));
}

void UnitTests::sphereAABBIntersectionTest14()
{
	Sphere sphere(glm::vec3(0.0f, 0.0f, 0.0f), 0.1f);
	AABB aabb(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(1.0f, 1.0f, 1.0f));

	bool answer = Intersect::intersect(sphere, aabb);

	print("Intersection of sphere and AABB (14) ", (answer == true));
}

void UnitTests::sphereAABBIntersectionTest15()
{
	Sphere sphere(glm::vec3(1.0, 1.0f, 1.0f), 0.5f);
	AABB aabb(glm::vec3(1.5f, 1.5f, 0.0f), glm::vec3(1.0f, 1.0f, 1.0f));

	bool answer = Intersect::intersect(sphere, aabb);

	print("Intersection of sphere and AABB (15) ", (answer == true));
}

void UnitTests::sphereAABBIntersectionTest16()
{
	Sphere sphere(glm::vec3(-2.0f, 3.0f, 1.1f), 2.3f);
	AABB aabb(glm::vec3(5.0f, 10.0f, 4.0f), glm::vec3(1.2f, -2.2f, 4.2f));

	bool answer = Intersect::intersect(sphere, aabb);

	print("Intersection of sphere and AABB (16) ", (answer == false));
}

void UnitTests::sphereAABBIntersectionTest17()
{
	Sphere sphere(glm::vec3(-2.0f, 0.0f, 0.0f), 0.4f);
	AABB aabb(glm::vec3(1.0f, -1.0f, 0.0f), glm::vec3(0.2f, 0.2f, 0.2f));

	bool answer = Intersect::intersect(sphere, aabb);

	print("Intersection of sphere and AABB (17) ", (answer == false));
}

void UnitTests::sphereAABBIntersectionTest18()
{
	Sphere sphere(glm::vec3(-1.0f, -1.0f, -1.0f), 0.4f);
	AABB aabb(glm::vec3(1.0f, 2.0f, 3.0f), glm::vec3(1.2f, 0.2f, 0.8f));

	bool answer = Intersect::intersect(sphere, aabb);

	print("Intersection of sphere and AABB (18) ", (answer == false));
}

void UnitTests::sphereSphereIntersectionTest0()
{
	Sphere sphere1(glm::vec3(0.0f, 0.0f, 0.0f), 0.75f);
	Sphere sphere2(glm::vec3(1.0f, 0.0f, 0.0f), 0.75f);

	bool answer = Intersect::intersect(sphere1, sphere2);

	print("Intersection of sphere and sphere (0) ", (answer == true));
}

void UnitTests::sphereSphereIntersectionTest1()
{
	Sphere sphere1(glm::vec3(0.0f, 0.0f, 0.0f), 0.75f);
	Sphere sphere2(glm::vec3(0.0f, 1.0f, 0.0f), 0.75f);

	bool answer = Intersect::intersect(sphere1, sphere2);

	print("Intersection of sphere and sphere (1) ", (answer == true));
}

void UnitTests::sphereSphereIntersectionTest2()
{
	Sphere sphere1(glm::vec3(0.0f, 0.0f, 0.0f), 0.75f);
	Sphere sphere2(glm::vec3(0.0f, 0.0f, 1.0f), 0.75f);

	bool answer = Intersect::intersect(sphere1, sphere2);

	print("Intersection of sphere and sphere (2) ", (answer == true));
}

void UnitTests::sphereSphereIntersectionTest3()
{
	Sphere sphere1(glm::vec3(0.0f, 0.0f, 0.0f), 0.75f);
	Sphere sphere2(glm::vec3(-1.0f, 0.0f, 0.0f), 0.75f);

	bool answer = Intersect::intersect(sphere1, sphere2);

	print("Intersection of sphere and sphere (3) ", (answer == true));
}

void UnitTests::sphereSphereIntersectionTest4()
{
	Sphere sphere1(glm::vec3(0.0f, 0.0f, 0.0f), 0.75f);
	Sphere sphere2(glm::vec3(0.0f, -1.0f, 0.0f), 0.75f);

	bool answer = Intersect::intersect(sphere1, sphere2);

	print("Intersection of sphere and sphere (4) ", (answer == true));
}

void UnitTests::sphereSphereIntersectionTest5()
{
	Sphere sphere1(glm::vec3(0.0f, 0.0f, 0.0f), 0.75f);
	Sphere sphere2(glm::vec3(0.0f, 0.0f, -1.0f), 0.75f);

	bool answer = Intersect::intersect(sphere1, sphere2);

	print("Intersection of sphere and sphere (5) ", (answer == true));
}

void UnitTests::sphereSphereIntersectionTest6()
{
	Sphere sphere1(glm::vec3(0.0f, 0.0f, 0.0f), 1.0f);
	Sphere sphere2(glm::vec3(2.0f, 0.0f, 0.0f), 1.0f);

	bool answer = Intersect::intersect(sphere1, sphere2);

	print("Intersection of sphere and sphere (6) ", (answer == true));
}

void UnitTests::sphereSphereIntersectionTest7()
{
	Sphere sphere1(glm::vec3(1.0f, 1.0f, 1.0f), 0.25f);
	Sphere sphere2(glm::vec3(-2.0f, 0.0f, 0.0f), 1.0f);

	bool answer = Intersect::intersect(sphere1, sphere2);

	print("Intersection of sphere and sphere (7) ", (answer == false));
}

void UnitTests::sphereSphereIntersectionTest8()
{
	Sphere sphere1(glm::vec3(1.0f, -2.0f, 3.0f), 0.25f);
	Sphere sphere2(glm::vec3(-20.0f, 1.0f, 5.0f), 0.45f);

	bool answer = Intersect::intersect(sphere1, sphere2);

	print("Intersection of sphere and sphere (8) ", (answer == false));
}

void UnitTests::sphereSphereIntersectionTest9()
{
	Sphere sphere1(glm::vec3(1.0f, -2.0f, 3.0f), 0.5f);
	Sphere sphere2(glm::vec3(1.0f, -2.0f, 3.0f), 0.5f);

	bool answer = Intersect::intersect(sphere1, sphere2);

	print("Intersection of sphere and sphere (9) ", (answer == true));
}

void UnitTests::sphereSphereIntersectionTest10()
{
	Sphere sphere1(glm::vec3(1.0f, -2.0f, 3.0f), 0.5f);
	Sphere sphere2(glm::vec3(1.0f, -2.0f, 3.0f), 0.25f);

	bool answer = Intersect::intersect(sphere1, sphere2);

	print("Intersection of sphere and sphere (10) ", (answer == true));
}

void UnitTests::sphereSphereIntersectionTest11()
{
	Sphere sphere1(glm::vec3(1.0f, -2.0f, 3.0f), 0.25f);
	Sphere sphere2(glm::vec3(1.0f, -2.0f, 3.0f), 0.5f);

	bool answer = Intersect::intersect(sphere1, sphere2);

	print("Intersection of sphere and sphere (11) ", (answer == true));
}

void UnitTests::sphereSphereIntersectionTest12()
{
	Sphere sphere1(glm::vec3(1.0f, -2.0f, 3.0f), 2.5f);
	Sphere sphere2(glm::vec3(1.1f, -1.8f, 3.2f), 0.25f);

	bool answer = Intersect::intersect(sphere1, sphere2);

	print("Intersection of sphere and sphere (12) ", (answer == true));
}

















void UnitTests::sphereFrustumIntersectionTest0()
{
	Frustum frustum;
	frustum.mFov = 45.0f;
	frustum.mAspectRatio = 1.0f;
	frustum.mNearPlane = 0.1f;
	frustum.mFarPlane = 250.0f;

	glm::vec3 position = glm::vec3(0, 2, -10);
	glm::vec3 front = glm::vec3(0, 0, 1);
	glm::vec3 up = glm::vec3(0, 1, 0);
	glm::vec3 right = glm::vec3(1, 0, 0);

	frustum.computePlanes(position, front, up, right);
	
	Sphere sphere;
	sphere.mCentre = glm::vec3(-0.04142, 2.0414, -9.8999);
	sphere.mRadius = 0.01f;

	bool answer = Intersect::intersect(sphere, frustum);

	print("Intersection of sphere and frustum (0) ", (answer == true));
}

void UnitTests::sphereFrustumIntersectionTest1()
{
	Frustum frustum;
	frustum.mFov = 45.0f;
	frustum.mAspectRatio = 1.0f;
	frustum.mNearPlane = 0.1f;
	frustum.mFarPlane = 250.0f;

	glm::vec3 position = glm::vec3(0, 2, -10);
	glm::vec3 front = glm::vec3(0, 0, 1);
	glm::vec3 up = glm::vec3(0, 1, 0);
	glm::vec3 right = glm::vec3(1, 0, 0);

	frustum.computePlanes(position, front, up, right);

	Sphere sphere;
	sphere.mCentre = glm::vec3(0.04142, 2.0414, -9.8999);
	sphere.mRadius = 0.01f;

	bool answer = Intersect::intersect(sphere, frustum);

	print("Intersection of sphere and frustum (1) ", (answer == true));
}

void UnitTests::sphereFrustumIntersectionTest2()
{
	Frustum frustum;
	frustum.mFov = 45.0f;
	frustum.mAspectRatio = 1.0f;
	frustum.mNearPlane = 0.1f;
	frustum.mFarPlane = 250.0f;

	glm::vec3 position = glm::vec3(0, 2, -10);
	glm::vec3 front = glm::vec3(0, 0, 1);
	glm::vec3 up = glm::vec3(0, 1, 0);
	glm::vec3 right = glm::vec3(1, 0, 0);

	frustum.computePlanes(position, front, up, right);

	Sphere sphere;
	sphere.mCentre = glm::vec3(-0.04142, 1.9585, -9.8999);
	sphere.mRadius = 0.01f;

	bool answer = Intersect::intersect(sphere, frustum);

	print("Intersection of sphere and frustum (2) ", (answer == true));
}

void UnitTests::sphereFrustumIntersectionTest3()
{
	Frustum frustum;
	frustum.mFov = 45.0f;
	frustum.mAspectRatio = 1.0f;
	frustum.mNearPlane = 0.1f;
	frustum.mFarPlane = 250.0f;

	glm::vec3 position = glm::vec3(0, 2, -10);
	glm::vec3 front = glm::vec3(0, 0, 1);
	glm::vec3 up = glm::vec3(0, 1, 0);
	glm::vec3 right = glm::vec3(1, 0, 0);

	frustum.computePlanes(position, front, up, right);

	Sphere sphere;
	sphere.mCentre = glm::vec3(0.04142, 1.9585, -9.8999);
	sphere.mRadius = 0.01f;

	bool answer = Intersect::intersect(sphere, frustum);

	print("Intersection of sphere and frustum (3) ", (answer == true));
}

void UnitTests::sphereFrustumIntersectionTest4()
{
	Frustum frustum;
	frustum.mFov = 45.0f;
	frustum.mAspectRatio = 1.0f;
	frustum.mNearPlane = 0.1f;
	frustum.mFarPlane = 250.0f;

	glm::vec3 position = glm::vec3(0, 2, -10);
	glm::vec3 front = glm::vec3(0, 0, 1);
	glm::vec3 up = glm::vec3(0, 1, 0);
	glm::vec3 right = glm::vec3(1, 0, 0);

	frustum.computePlanes(position, front, up, right);

	Sphere sphere;
	sphere.mCentre = glm::vec3(-103, 105, 240);
	sphere.mRadius = 1.0f;

	bool answer = Intersect::intersect(sphere, frustum);

	print("Intersection of sphere and frustum (4) ", (answer == true));
}

void UnitTests::sphereFrustumIntersectionTest5()
{
	Frustum frustum;
	frustum.mFov = 45.0f;
	frustum.mAspectRatio = 1.0f;
	frustum.mNearPlane = 0.1f;
	frustum.mFarPlane = 250.0f;

	glm::vec3 position = glm::vec3(0, 2, -10);
	glm::vec3 front = glm::vec3(0, 0, 1);
	glm::vec3 up = glm::vec3(0, 1, 0);
	glm::vec3 right = glm::vec3(1, 0, 0);

	frustum.computePlanes(position, front, up, right);

	Sphere sphere;
	sphere.mCentre = glm::vec3(103, 105, 240);
	sphere.mRadius = 1.0f;

	bool answer = Intersect::intersect(sphere, frustum);

	print("Intersection of sphere and frustum (5) ", (answer == true));
}

void UnitTests::sphereFrustumIntersectionTest6()
{
	Frustum frustum;
	frustum.mFov = 45.0f;
	frustum.mAspectRatio = 1.0f;
	frustum.mNearPlane = 0.1f;
	frustum.mFarPlane = 250.0f;

	glm::vec3 position = glm::vec3(0, 2, -10);
	glm::vec3 front = glm::vec3(0, 0, 1);
	glm::vec3 up = glm::vec3(0, 1, 0);
	glm::vec3 right = glm::vec3(1, 0, 0);

	frustum.computePlanes(position, front, up, right);

	Sphere sphere;
	sphere.mCentre = glm::vec3(-103, -101, 240);
	sphere.mRadius = 1.0f;

	bool answer = Intersect::intersect(sphere, frustum);

	print("Intersection of sphere and frustum (6) ", (answer == true));
}

void UnitTests::sphereFrustumIntersectionTest7()
{
	Frustum frustum;
	frustum.mFov = 45.0f;
	frustum.mAspectRatio = 1.0f;
	frustum.mNearPlane = 0.1f;
	frustum.mFarPlane = 250.0f;

	glm::vec3 position = glm::vec3(0, 2, -10);
	glm::vec3 front = glm::vec3(0, 0, 1);
	glm::vec3 up = glm::vec3(0, 1, 0);
	glm::vec3 right = glm::vec3(1, 0, 0);

	frustum.computePlanes(position, front, up, right);

	Sphere sphere;
	sphere.mCentre = glm::vec3(103, -101, 240);
	sphere.mRadius = 1.0f;

	bool answer = Intersect::intersect(sphere, frustum);

	print("Intersection of sphere and frustum (7) ", (answer == true));
}

void UnitTests::sphereFrustumIntersectionTest8()
{
	Frustum frustum;
	frustum.mFov = 45.0f;
	frustum.mAspectRatio = 1.0f;
	frustum.mNearPlane = 0.1f;
	frustum.mFarPlane = 250.0f;

	glm::vec3 position = glm::vec3(0, 2, -10);
	glm::vec3 front = glm::vec3(0, 0, 1);
	glm::vec3 up = glm::vec3(0, 1, 0);
	glm::vec3 right = glm::vec3(1, 0, 0);

	frustum.computePlanes(position, front, up, right);

	Sphere sphere;
	sphere.mCentre = glm::vec3(0, 2, 0);
	sphere.mRadius = 1.0f;

	bool answer = Intersect::intersect(sphere, frustum);

	print("Intersection of sphere and frustum (8) ", (answer == true));
}

void UnitTests::sphereFrustumIntersectionTest9()
{
	Frustum frustum;
	frustum.mFov = 45.0f;
	frustum.mAspectRatio = 1.0f;
	frustum.mNearPlane = 0.1f;
	frustum.mFarPlane = 250.0f;

	glm::vec3 position = glm::vec3(0, 2, -10);
	glm::vec3 front = glm::vec3(0, 0, 1);
	glm::vec3 up = glm::vec3(0, 1, 0);
	glm::vec3 right = glm::vec3(1, 0, 0);

	frustum.computePlanes(position, front, up, right);

	Sphere sphere;
	sphere.mCentre = glm::vec3(0, 2, 0);
	sphere.mRadius = 1000.0f;

	bool answer = Intersect::intersect(sphere, frustum);

	print("Intersection of sphere and frustum (9) ", (answer == true));
}

void UnitTests::sphereFrustumIntersectionTest10()
{
	Frustum frustum;
	frustum.mFov = 45.0f;
	frustum.mAspectRatio = 1.0f;
	frustum.mNearPlane = 0.1f;
	frustum.mFarPlane = 250.0f;

	glm::vec3 position = glm::vec3(0, 2, -10);
	glm::vec3 front = glm::vec3(0, 0, 1);
	glm::vec3 up = glm::vec3(0, 1, 0);
	glm::vec3 right = glm::vec3(1, 0, 0);

	frustum.computePlanes(position, front, up, right);

	Sphere sphere;
	sphere.mCentre = glm::vec3(-0.06, 2.0414, -10);
	sphere.mRadius = 0.01f;

	bool answer = Intersect::intersect(sphere, frustum);

	print("Intersection of sphere and frustum (10) ", (answer == false));
}

void UnitTests::sphereFrustumIntersectionTest11()
{
	Frustum frustum;
	frustum.mFov = 45.0f;
	frustum.mAspectRatio = 1.0f;
	frustum.mNearPlane = 0.1f;
	frustum.mFarPlane = 250.0f;

	glm::vec3 position = glm::vec3(0, 2, -10);
	glm::vec3 front = glm::vec3(0, 0, 1);
	glm::vec3 up = glm::vec3(0, 1, 0);
	glm::vec3 right = glm::vec3(1, 0, 0);

	frustum.computePlanes(position, front, up, right);

	Sphere sphere;
	sphere.mCentre = glm::vec3(-105, 106, 240);
	sphere.mRadius = 1.0f;

	bool answer = Intersect::intersect(sphere, frustum);

	print("Intersection of sphere and frustum (11) ", (answer == false));
}

void UnitTests::sphereFrustumIntersectionTest12()
{
	Frustum frustum;
	frustum.mFov = 45.0f;
	frustum.mAspectRatio = 1.0f;
	frustum.mNearPlane = 0.1f;
	frustum.mFarPlane = 250.0f;

	glm::vec3 position = glm::vec3(0, 2, -10);
	glm::vec3 front = glm::vec3(0, 0, 1);
	glm::vec3 up = glm::vec3(0, 1, 0);
	glm::vec3 right = glm::vec3(1, 0, 0);

	frustum.computePlanes(position, front, up, right);

	Sphere sphere;
	sphere.mCentre = glm::vec3(-104, 106, 250);
	sphere.mRadius = 2.0f;

	bool answer = Intersect::intersect(sphere, frustum);

	print("Intersection of sphere and frustum (12) ", (answer == false));
}

void UnitTests::sphereFrustumIntersectionTest13()
{
	Frustum frustum;
	frustum.mFov = 45.0f;
	frustum.mAspectRatio = 1.0f;
	frustum.mNearPlane = 0.1f;
	frustum.mFarPlane = 250.0f;

	glm::vec3 position = glm::vec3(0, 2, -10);
	glm::vec3 front = glm::vec3(0, 0, 1);
	glm::vec3 up = glm::vec3(0, 1, 0);
	glm::vec3 right = glm::vec3(1, 0, 0);

	frustum.computePlanes(position, front, up, right);

	Sphere sphere;
	sphere.mCentre = glm::vec3(0, 2, -14);
	sphere.mRadius = 2.0f;

	bool answer = Intersect::intersect(sphere, frustum);

	print("Intersection of sphere and frustum (13) ", (answer == false));
}















void UnitTests::aabbAABBIntersectionTest0()
{
	AABB aabb1(glm::vec3(0.0, 0.0f, 0.0f), glm::vec3(1.0f, 1.0f, 1.0f));
	AABB aabb2(glm::vec3(1.0f, 0.0f, 0.0f), glm::vec3(1.5f, 1.5f, 1.5f));

	bool answer = Intersect::intersect(aabb1, aabb2);

	print("Intersection of AABB and AABB (0) ", (answer == true));
}

void UnitTests::aabbAABBIntersectionTest1()
{
	AABB aabb1(glm::vec3(0.0, 0.0f, 0.0f), glm::vec3(1.0f, 1.0f, 1.0f));
	AABB aabb2(glm::vec3(-1.0f, 0.0f, 0.0f), glm::vec3(1.5f, 1.5f, 1.5f));

	bool answer = Intersect::intersect(aabb1, aabb2);

	print("Intersection of AABB and AABB (1) ", (answer == true));
}

void UnitTests::aabbAABBIntersectionTest2()
{
	AABB aabb1(glm::vec3(0.0, 0.0f, 0.0f), glm::vec3(1.0f, 1.0f, 1.0f));
	AABB aabb2(glm::vec3(0.0f, 1.0f, 0.0f), glm::vec3(1.5f, 1.5f, 1.5f));

	bool answer = Intersect::intersect(aabb1, aabb2);

	print("Intersection of AABB and AABB (2) ", (answer == true));
}

void UnitTests::aabbAABBIntersectionTest3()
{
	AABB aabb1(glm::vec3(0.0, 0.0f, 0.0f), glm::vec3(1.0f, 1.0f, 1.0f));
	AABB aabb2(glm::vec3(0.0f, -1.0f, 0.0f), glm::vec3(1.5f, 1.5f, 1.5f));

	bool answer = Intersect::intersect(aabb1, aabb2);

	print("Intersection of AABB and AABB (3) ", (answer == true));
}

void UnitTests::aabbAABBIntersectionTest4()
{
	AABB aabb1(glm::vec3(0.0, 0.0f, 0.0f), glm::vec3(1.0f, 1.0f, 1.0f));
	AABB aabb2(glm::vec3(0.0f, 0.0f, 1.0f), glm::vec3(1.5f, 1.5f, 1.5f));

	bool answer = Intersect::intersect(aabb1, aabb2);

	print("Intersection of AABB and AABB (4) ", (answer == true));
}

void UnitTests::aabbAABBIntersectionTest5()
{
	AABB aabb1(glm::vec3(0.0, 0.0f, 0.0f), glm::vec3(1.0f, 1.0f, 1.0f));
	AABB aabb2(glm::vec3(0.0f, 0.0f, -1.0f), glm::vec3(1.5f, 1.5f, 1.5f));

	bool answer = Intersect::intersect(aabb1, aabb2);

	print("Intersection of AABB and AABB (5) ", (answer == true));
}

void UnitTests::aabbAABBIntersectionTest6()
{
	AABB aabb1(glm::vec3(0.0, 0.0f, 0.0f), glm::vec3(1.0f, 1.0f, 1.0f));
	AABB aabb2(glm::vec3(1.0f, 0.0f, 1.0f), glm::vec3(1.0f, 1.0f, 1.0f));

	bool answer = Intersect::intersect(aabb1, aabb2);

	print("Intersection of AABB and AABB (6) ", (answer == true));
}

void UnitTests::aabbAABBIntersectionTest7()
{
	AABB aabb1(glm::vec3(0.0, 0.0f, 0.0f), glm::vec3(1.0f, 1.0f, 1.0f));
	AABB aabb2(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.5f, 0.5f, 0.5f));

	bool answer = Intersect::intersect(aabb1, aabb2);

	print("Intersection of AABB and AABB (7) ", (answer == true));
}

void UnitTests::aabbAABBIntersectionTest8()
{
	AABB aabb1(glm::vec3(0.0, 0.0f, 0.0f), glm::vec3(1.0f, 1.0f, 1.0f));
	AABB aabb2(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.5f, 0.5f, 0.5f));

	bool answer = Intersect::intersect(aabb1, aabb2);

	print("Intersection of AABB and AABB (8) ", (answer == true));
}

void UnitTests::aabbAABBIntersectionTest9()
{
	AABB aabb1(glm::vec3(0.0, 0.0f, 0.0f), glm::vec3(2.0f, 1.0f, -1.0f));
	AABB aabb2(glm::vec3(3.0f, 4.0f, 5.0f), glm::vec3(0.5f, 0.5f, 0.5f));

	bool answer = Intersect::intersect(aabb1, aabb2);

	print("Intersection of AABB and AABB (9) ", (answer == false));
}

void UnitTests::aabbAABBIntersectionTest10()
{
	AABB aabb1(glm::vec3(-1.0, 0.5f, -0.7f), glm::vec3(2.0f, 1.0f, -1.0f));
	AABB aabb2(glm::vec3(3.0f, 4.0f, 5.0f), glm::vec3(0.5f, 0.5f, 0.5f));

	bool answer = Intersect::intersect(aabb1, aabb2);

	print("Intersection of AABB and AABB (10) ", (answer == false));
}
