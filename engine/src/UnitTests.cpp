#include <iostream>

#include "../include/UnitTests.h"


#include "../include/glm/glm.hpp"

#include "../include/core/Log.h"
#include "../include/core/PoolAllocator.h"
#include "../include/core/Geometry.h"
#include "../include/core/Physics.h"

#include "../include/components/Transform.h"

using namespace PhysicsEngine;

void UnitTests::print(const std::string testMessage, bool testPassed)
{
	if (testPassed) {
		std::string message = "CORE UNIT TEST: " + testMessage + " PASSED\n";
		Log::info(message.c_str());
	}
	else {
		std::string message = "CORE UNIT TEST: " + testMessage + " FAILED\n";
		Log::info(message.c_str());
	}
}

void UnitTests::run()
{
	poolAllocatorTest0();
	poolAllocatorTest1();
	poolAllocatorTest2();
	poolAllocatorTest3();
	poolAllocatorTest4();

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

	rayBoundsIntersectionTest0();
	rayBoundsIntersectionTest1();
	rayBoundsIntersectionTest2();
	rayBoundsIntersectionTest3();
	rayBoundsIntersectionTest4();
	rayBoundsIntersectionTest5();
	rayBoundsIntersectionTest6();
	rayBoundsIntersectionTest7();
	rayBoundsIntersectionTest8();
	rayBoundsIntersectionTest9();
	rayBoundsIntersectionTest10();
	rayBoundsIntersectionTest11();
	rayBoundsIntersectionTest12();
	rayBoundsIntersectionTest13();
	rayBoundsIntersectionTest14();
	rayBoundsIntersectionTest15();
	rayBoundsIntersectionTest16();
	rayBoundsIntersectionTest17();
	rayBoundsIntersectionTest18();

	sphereBoundsIntersectionTest0();
	sphereBoundsIntersectionTest1();
	sphereBoundsIntersectionTest2();
	sphereBoundsIntersectionTest3();
	sphereBoundsIntersectionTest4();
	sphereBoundsIntersectionTest5();
	sphereBoundsIntersectionTest6();
	sphereBoundsIntersectionTest7();
	sphereBoundsIntersectionTest8();
	sphereBoundsIntersectionTest9();
	sphereBoundsIntersectionTest10();
	sphereBoundsIntersectionTest11();
	sphereBoundsIntersectionTest12();
	sphereBoundsIntersectionTest13();
	sphereBoundsIntersectionTest14();
	sphereBoundsIntersectionTest15();
	sphereBoundsIntersectionTest16();
	sphereBoundsIntersectionTest17();
	sphereBoundsIntersectionTest18();

	sphereSphereIntersectionTest0();
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
	sphereSphereIntersectionTest12();

	boundsBoundsIntersectionTest0();
	boundsBoundsIntersectionTest1();
	boundsBoundsIntersectionTest2();
	boundsBoundsIntersectionTest3();
	boundsBoundsIntersectionTest4();
	boundsBoundsIntersectionTest5();
	boundsBoundsIntersectionTest6();
	boundsBoundsIntersectionTest7();
	boundsBoundsIntersectionTest8();
	boundsBoundsIntersectionTest9();
	boundsBoundsIntersectionTest10();
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

void UnitTests::raySphereIntersectionTest0()
{
	Ray ray(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(1.0f, 0.0f, 0.0f));
	Sphere sphere(glm::vec3(10.0f, 0.0f, 0.0f), 1.0f);

	bool answer = Geometry::intersect(ray, sphere);

	print("Intersection of ray and sphere (0) ", (answer == true));
}

void UnitTests::raySphereIntersectionTest1()
{
	Ray ray(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
	Sphere sphere(glm::vec3(0.0f, 10.0f, 0.0f), 1.0f);

	bool answer = Geometry::intersect(ray, sphere);

	print("Intersection of ray and sphere (1) ", (answer == true));
}

void UnitTests::raySphereIntersectionTest2()
{
	Ray ray(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f));
	Sphere sphere(glm::vec3(0.0f, 0.0f, 10.0f), 1.0f);

	bool answer = Geometry::intersect(ray, sphere);

	print("Intersection of ray and sphere (2) ", (answer == true));
}

void UnitTests::raySphereIntersectionTest3()
{
	Ray ray(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(-1.0f, 0.0f, 0.0f));
	Sphere sphere(glm::vec3(-10.0f, 0.0f, 0.0f), 1.0f);

	bool answer = Geometry::intersect(ray, sphere);

	print("Intersection of ray and sphere (3) ", (answer == true));
}

void UnitTests::raySphereIntersectionTest4()
{
	Ray ray(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, -1.0f, 0.0f));
	Sphere sphere(glm::vec3(0.0f, -10.0f, 0.0f), 1.0f);

	bool answer = Geometry::intersect(ray, sphere);

	print("Intersection of ray and sphere (4) ", (answer == true));
}

void UnitTests::raySphereIntersectionTest5()
{
	Ray ray(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, -1.0f));
	Sphere sphere(glm::vec3(0.0f, 0.0f, -10.0f), 1.0f);

	bool answer = Geometry::intersect(ray, sphere);

	print("Intersection of ray and sphere (5) ", (answer == true));
}

void UnitTests::raySphereIntersectionTest6()
{
	Ray ray(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(1.0f, 1.0f, 1.0f));
	Sphere sphere(glm::vec3(10.0f, 10.0f, 10.0f), 1.0f);

	bool answer = Geometry::intersect(ray, sphere);

	print("Intersection of ray and sphere (6) ", (answer == true));
}

void UnitTests::raySphereIntersectionTest7()
{
	Ray ray(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(-1.0f, -1.0f, -1.0f));
	Sphere sphere(glm::vec3(-10.0f, -10.0f, -10.0f), 1.0f);

	bool answer = Geometry::intersect(ray, sphere);

	print("Intersection of ray and sphere (7) ", (answer == true));
}

void UnitTests::raySphereIntersectionTest8()
{
	Ray ray(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(10.0f, 0.5f, 0.5f));
	Sphere sphere(glm::vec3(10.0f, 0.0f, 0.0f), 1.0f);

	bool answer = Geometry::intersect(ray, sphere);

	print("Intersection of ray and sphere (8) ", (answer == true));
}

void UnitTests::raySphereIntersectionTest9()
{
	Ray ray(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(10.0f, 1.0f, 0.0f));
	Sphere sphere(glm::vec3(10.0f, 0.0f, 0.0f), 1.0f);

	bool answer = Geometry::intersect(ray, sphere);

	print("Intersection of ray and sphere (9) ", (answer == true));
}

void UnitTests::raySphereIntersectionTest10()
{
	Ray ray(glm::vec3(1.0f, 1.0f, 1.0f), glm::vec3(1.0f, 0.0f, 0.0f));
	Sphere sphere(glm::vec3(10.0f, 0.0f, 0.0f), 2.0f);

	bool answer = Geometry::intersect(ray, sphere);

	print("Intersection of ray and sphere (10) ", (answer == true));
}

void UnitTests::raySphereIntersectionTest11()
{
	Ray ray(glm::vec3(-1.0f, -1.0f, -1.0f), glm::vec3(1.0f, 0.0f, 0.0f));
	Sphere sphere(glm::vec3(10.0f, 0.0f, 0.0f), 2.0f);

	bool answer = Geometry::intersect(ray, sphere);

	print("Intersection of ray and sphere (11) ", (answer == true));
}

void UnitTests::raySphereIntersectionTest12()
{
	Ray ray(glm::vec3(1.0f, 2.0f, 3.0f), glm::vec3(-1.0f, -2.0f, -3.0f));
	Sphere sphere(glm::vec3(0.0f, 0.0f, 0.0f), 1.0f);

	bool answer = Geometry::intersect(ray, sphere);

	print("Intersection of ray and sphere (12) ", (answer == true));
}

void UnitTests::raySphereIntersectionTest13()
{
	Ray ray(glm::vec3(-1.0f, -2.0f, -3.0f), glm::vec3(1.0f, 2.0f, 3.0f));
	Sphere sphere(glm::vec3(0.0f, 0.0f, 0.0f), 1.0f);

	bool answer = Geometry::intersect(ray, sphere);

	print("Intersection of ray and sphere (13) ", (answer == true));
}

void UnitTests::raySphereIntersectionTest14()
{
	Ray ray(glm::vec3(1.0f, 2.0f, 3.0f), glm::vec3(-1.0f, -2.0f, -3.0f));
	Sphere sphere(glm::vec3(0.2f, 0.3f, 0.1f), 2.0f);

	bool answer = Geometry::intersect(ray, sphere);

	print("Intersection of ray and sphere (14) ", (answer == true));
}

void UnitTests::raySphereIntersectionTest15()
{
	Ray ray(glm::vec3(-1.0f, -2.0f, -3.0f), glm::vec3(1.0f, 2.0f, 3.0f));
	Sphere sphere(glm::vec3(0.2f, 0.3f, 0.1f), 2.0f);

	bool answer = Geometry::intersect(ray, sphere);

	print("Intersection of ray and sphere (15) ", (answer == true));
}

void UnitTests::raySphereIntersectionTest16()
{
	Ray ray(glm::vec3(-1.0f, 2.0f, -3.0f), glm::vec3(4.0f, -2.0f, 3.0f));
	Sphere sphere(glm::vec3(-1.0f, 2.0f, -3.0f), 0.5f);

	bool answer = Geometry::intersect(ray, sphere);

	print("Intersection of ray and sphere (16) ", (answer == true));
}

void UnitTests::raySphereIntersectionTest17()
{
	Ray ray(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(1.0f, 0.0f, 0.0f));
	Sphere sphere(glm::vec3(-10.0f, 0.0f, 0.0f), 1.0f);

	bool answer = Geometry::intersect(ray, sphere);

	print("Intersection of ray and sphere (17) ", (answer == false));
}

void UnitTests::raySphereIntersectionTest18()
{
	Ray ray(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
	Sphere sphere(glm::vec3(0.0f, -10.0f, 0.0f), 1.0f);

	bool answer = Geometry::intersect(ray, sphere);

	print("Intersection of ray and sphere (18) ", (answer == false));
}

void UnitTests::raySphereIntersectionTest19()
{
	Ray ray(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f));
	Sphere sphere(glm::vec3(0.0f, 0.0f, -10.0f), 1.0f);

	bool answer = Geometry::intersect(ray, sphere);

	print("Intersection of ray and sphere (19) ", (answer == false));
}

void UnitTests::raySphereIntersectionTest20()
{
	Ray ray(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(1.0f, 1.0f, 1.0f));
	Sphere sphere(glm::vec3(-10.0f, -10.0f, -10.0f), 1.0f);

	bool answer = Geometry::intersect(ray, sphere);

	print("Intersection of ray and sphere (20) ", (answer == false));
}

void UnitTests::raySphereIntersectionTest21()
{
	Ray ray(glm::vec3(2.0f, 0.0f, 0.0f), glm::vec3(1.0f, 0.0f, 0.0f));
	Sphere sphere(glm::vec3(1.0f, 0.0f, 0.0f), 0.5f);

	bool answer = Geometry::intersect(ray, sphere);

	print("Intersection of ray and sphere (21) ", (answer == false));
}

void UnitTests::raySphereIntersectionTest22()
{
	Ray ray(glm::vec3(1.0f, 1.0f, 1.0f), glm::vec3(1.0f, 0.0f, 0.0f));
	Sphere sphere(glm::vec3(10.0f, 0.0f, 0.0f), 1.0f);

	bool answer = Geometry::intersect(ray, sphere);

	print("Intersection of ray and sphere (22) ", (answer == false));
}

void UnitTests::rayBoundsIntersectionTest0()
{
	Ray ray(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(1.0f, 0.0f, 0.0f));
	Bounds bounds(glm::vec3(10.0f, 0.0f, 0.0f), glm::vec3(1.0f, 1.0f, 1.0f));

	bool answer = Geometry::intersect(ray, bounds);

	print("Intersection of ray and bounds (0) ", (answer == true));
}

void UnitTests::rayBoundsIntersectionTest1()
{
	Ray ray(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
	Bounds bounds(glm::vec3(0.0f, 10.0f, 0.0f), glm::vec3(1.0f, 1.0f, 1.0f));

	bool answer = Geometry::intersect(ray, bounds);

	print("Intersection of ray and bounds (1) ", (answer == true));
}

void UnitTests::rayBoundsIntersectionTest2()
{
	Ray ray(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f));
	Bounds bounds(glm::vec3(0.0f, 0.0f, 10.0f), glm::vec3(1.0f, 1.0f, 1.0f));

	bool answer = Geometry::intersect(ray, bounds);

	print("Intersection of ray and bounds (2) ", (answer == true));
}

void UnitTests::rayBoundsIntersectionTest3()
{
	Ray ray(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(-1.0f, 0.0f, 0.0f));
	Bounds bounds(glm::vec3(-10.0f, 0.0f, 0.0f), glm::vec3(1.0f, 1.0f, 1.0f));

	bool answer = Geometry::intersect(ray, bounds);

	print("Intersection of ray and bounds (3) ", (answer == true));
}

void UnitTests::rayBoundsIntersectionTest4()
{
	Ray ray(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, -1.0f, 0.0f));
	Bounds bounds(glm::vec3(0.0f, -10.0f, 0.0f), glm::vec3(1.0f, 1.0f, 1.0f));

	bool answer = Geometry::intersect(ray, bounds);

	print("Intersection of ray and bounds (4) ", (answer == true));
}

void UnitTests::rayBoundsIntersectionTest5()
{
	Ray ray(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, -1.0f));
	Bounds bounds(glm::vec3(0.0f, 0.0f, -10.0f), glm::vec3(1.0f, 1.0f, 1.0f));

	bool answer = Geometry::intersect(ray, bounds);

	print("Intersection of ray and bounds (5) ", (answer == true));
}

void UnitTests::rayBoundsIntersectionTest6()
{
	Ray ray(glm::vec3(1.0f, 1.0f, 1.0f), glm::vec3(1.0f, 0.0f, 0.0f));
	Bounds bounds(glm::vec3(2.0f, 0.0f, 0.0f), glm::vec3(2.001f, 2.0001f, 2.001f));

	bool answer = Geometry::intersect(ray, bounds);

	print("Intersection of ray and bounds (6) ", (answer == true));
}

void UnitTests::rayBoundsIntersectionTest7()
{
	Ray ray(glm::vec3(1.0f, 1.0f, 1.0f), glm::vec3(1.0f, 0.0f, 0.0f));
	Bounds bounds(glm::vec3(2.0f, 0.0f, 0.0f), glm::vec3(3.0f, 4.0f, 2.00001f));

	bool answer = Geometry::intersect(ray, bounds);

	print("Intersection of ray and bounds (7) ", (answer == true));
}

void UnitTests::rayBoundsIntersectionTest8()
{
	Ray ray(glm::vec3(0.0f, 1.0f, 0.0f), glm::vec3(-1.0f, -1.0f, -1.0f));
	Bounds bounds(glm::vec3(0.0f, -2.0f, 0.0f), glm::vec3(4.0f, 4.0f, 4.0f));

	bool answer = Geometry::intersect(ray, bounds);

	print("Intersection of ray and bounds (8) ", (answer == true));
}

void UnitTests::rayBoundsIntersectionTest9()
{
	Ray ray(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(1.0f, 0.0f, 0.0f));
	Bounds bounds(glm::vec3(-1.0f, 0.0f, 0.0f), glm::vec3(4.0f, 4.0f, 4.0f));

	bool answer = Geometry::intersect(ray, bounds);

	print("Intersection of ray and bounds (9) ", (answer == true));
}

void UnitTests::rayBoundsIntersectionTest10()
{
	Ray ray(glm::vec3(-2.0f, 0.0f, 0.0f), glm::vec3(1.0f, 0.0f, 0.0f));
	Bounds bounds(glm::vec3(-1.0f, 0.0f, 0.0f), glm::vec3(4.0f, 4.0f, 4.0f));

	bool answer = Geometry::intersect(ray, bounds);

	print("Intersection of ray and bounds (10) ", (answer == true));
}

void UnitTests::rayBoundsIntersectionTest11()
{
	Ray ray(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(1.0f, 0.0f, 0.0f));
	Bounds bounds(glm::vec3(-1.0f, 0.0f, 0.0f), glm::vec3(1.0f, 1.0f, 1.0f));

	bool answer = Geometry::intersect(ray, bounds);

	print("Intersection of ray and bounds (11) ", (answer == false));
}

void UnitTests::rayBoundsIntersectionTest12()
{
	Ray ray(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
	Bounds bounds(glm::vec3(0.0f, -1.0f, 0.0f), glm::vec3(1.0f, 1.0f, 1.0f));

	bool answer = Geometry::intersect(ray, bounds);

	print("Intersection of ray and bounds (12) ", (answer == false));
}

void UnitTests::rayBoundsIntersectionTest13()
{
	Ray ray(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f));
	Bounds bounds(glm::vec3(0.0f, 0.0f, -1.0f), glm::vec3(1.0f, 1.0f, 1.0f));

	bool answer = Geometry::intersect(ray, bounds);

	print("Intersection of ray and bounds (13) ", (answer == false));
}

void UnitTests::rayBoundsIntersectionTest14()
{
	Ray ray(glm::vec3(1.0f, 1.0f, 1.0f), glm::vec3(0.5f, 0.4f, 1.3f));
	Bounds bounds(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(1.0f, 1.0f, 1.0f));

	bool answer = Geometry::intersect(ray, bounds);

	print("Intersection of ray and bounds (14) ", (answer == false));
}

void UnitTests::rayBoundsIntersectionTest15()
{
	Ray ray(glm::vec3(1.0f, 1.0f, 1.0f), glm::vec3(-0.5f, -0.4f, -1.3f));
	Bounds bounds(glm::vec3(2.0f, 2.0f, 2.0f), glm::vec3(1.0f, 1.0f, 1.0f));

	bool answer = Geometry::intersect(ray, bounds);

	print("Intersection of ray and bounds (15) ", (answer == false));
}

void UnitTests::rayBoundsIntersectionTest16()
{
	Ray ray(glm::vec3(1.0f, 1.0f, 1.0f), glm::vec3(1.0f, 0.0f, 0.0f));
	Bounds bounds(glm::vec3(2.0f, 0.0f, 0.0f), glm::vec3(2.0f, 2.0f, 2.0f));

	bool answer = Geometry::intersect(ray, bounds);

	print("Intersection of ray and bounds (16) ", (answer == true));
}

void UnitTests::rayBoundsIntersectionTest17()
{
	Ray ray(glm::vec3(1.0f, 1.0f, 1.0f), glm::vec3(1.0f, 0.0f, 0.0f));
	Bounds bounds(glm::vec3(2.0f, 0.0f, 0.0f), glm::vec3(3.0f, 4.0f, 2.0f));

	bool answer = Geometry::intersect(ray, bounds);

	print("Intersection of ray and bounds (17) ", (answer == true));
}

void UnitTests::rayBoundsIntersectionTest18()
{
	Ray ray(glm::vec3(-1.0f, 10.0f, 2.2f), glm::vec3(1.0f, 0.1f, -2.3f));
	Bounds bounds(glm::vec3(-1.015f, 9.85f, 1.9f), glm::vec3(2.8f, 1.234f, 2.1f));

	bool answer = Geometry::intersect(ray, bounds);

	print("Intersection of ray and bounds (18) ", (answer == true));
}


void UnitTests::sphereBoundsIntersectionTest0()
{
	Sphere sphere(glm::vec3(0.0f, 0.0f, 0.0f), 1.0f);
	Bounds bounds(glm::vec3(1.0f, 0.0f, 0.0f), glm::vec3(1.0f, 1.0f, 1.0f));

	bool answer = Geometry::intersect(sphere, bounds);

	print("Intersection of sphere and bounds (0) ", (answer == true));
}

void UnitTests::sphereBoundsIntersectionTest1()
{
	Sphere sphere(glm::vec3(0.0f, 0.0f, 0.0f), 1.0f);
	Bounds bounds(glm::vec3(1.0f, 0.0f, 0.0f), glm::vec3(1.0f, 1.0f, 1.0f));

	bool answer = Geometry::intersect(sphere, bounds);

	print("Intersection of sphere and bounds (1) ", (answer == true));
}

void UnitTests::sphereBoundsIntersectionTest2()
{
	Sphere sphere(glm::vec3(0.0f, 0.0f, 0.0f), 1.0f);
	Bounds bounds(glm::vec3(-1.0f, 0.0f, 0.0f), glm::vec3(1.0f, 1.0f, 1.0f));

	bool answer = Geometry::intersect(sphere, bounds);

	print("Intersection of sphere and bounds (2) ", (answer == true));
}

void UnitTests::sphereBoundsIntersectionTest3()
{
	Sphere sphere(glm::vec3(0.0f, 0.0f, 0.0f), 1.0f);
	Bounds bounds(glm::vec3(0.0f, 1.0f, 0.0f), glm::vec3(1.0f, 1.0f, 1.0f));

	bool answer = Geometry::intersect(sphere, bounds);

	print("Intersection of sphere and bounds (3) ", (answer == true));
}

void UnitTests::sphereBoundsIntersectionTest4()
{
	Sphere sphere(glm::vec3(0.0f, 0.0f, 0.0f), 1.0f);
	Bounds bounds(glm::vec3(0.0f, -1.0f, 0.0f), glm::vec3(1.0f, 1.0f, 1.0f));

	bool answer = Geometry::intersect(sphere, bounds);

	print("Intersection of sphere and bounds (4) ", (answer == true));
}

void UnitTests::sphereBoundsIntersectionTest5()
{
	Sphere sphere(glm::vec3(0.0f, 0.0f, 0.0f), 1.0f);
	Bounds bounds(glm::vec3(0.0f, 0.0f, 1.0f), glm::vec3(1.0f, 1.0f, 1.0f));

	bool answer = Geometry::intersect(sphere, bounds);

	print("Intersection of sphere and bounds (5) ", (answer == true));
}

void UnitTests::sphereBoundsIntersectionTest6()
{
	Sphere sphere(glm::vec3(0.0f, 0.0f, 0.0f), 1.0f);
	Bounds bounds(glm::vec3(0.0f, 0.0f, -1.0f), glm::vec3(1.0f, 1.0f, 1.0f));

	bool answer = Geometry::intersect(sphere, bounds);

	print("Intersection of sphere and bounds (6) ", (answer == true));
}

void UnitTests::sphereBoundsIntersectionTest7()
{
	Sphere sphere(glm::vec3(0.0f, 0.0f, 0.0f), 1.0f);
	Bounds bounds(glm::vec3(1.0f, 1.0f, 1.0f), glm::vec3(1.0f, 1.0f, 1.0f));

	bool answer = Geometry::intersect(sphere, bounds);

	print("Intersection of sphere and bounds (7) ", (answer == true));
}

void UnitTests::sphereBoundsIntersectionTest8()
{
	Sphere sphere(glm::vec3(0.0f, 0.0f, 0.0f), 1.0f);
	Bounds bounds(glm::vec3(-1.0f, -1.0f, -1.0f), glm::vec3(1.0f, 1.0f, 1.0f));

	bool answer = Geometry::intersect(sphere, bounds);

	print("Intersection of sphere and bounds (8) ", (answer == true));
}

void UnitTests::sphereBoundsIntersectionTest9()
{
	Sphere sphere(glm::vec3(0.0f, 0.0f, 0.0f), 1.0f);
	Bounds bounds(glm::vec3(-1.0f, 1.0f, -1.0f), glm::vec3(1.0f, 1.0f, 1.0f));

	bool answer = Geometry::intersect(sphere, bounds);

	print("Intersection of sphere and bounds (9) ", (answer == true));
}

void UnitTests::sphereBoundsIntersectionTest10()
{
	Sphere sphere(glm::vec3(0.0f, 0.0f, 0.0f), 1.0f);
	Bounds bounds(glm::vec3(1.0f, -1.0f, 1.0f), glm::vec3(1.0f, 1.0f, 1.0f));

	bool answer = Geometry::intersect(sphere, bounds);

	print("Intersection of sphere and bounds (10) ", (answer == true));
}

void UnitTests::sphereBoundsIntersectionTest11()
{
	Sphere sphere(glm::vec3(0.0f, 0.0f, 0.0f), 1.0f);
	Bounds bounds(glm::vec3(1.0f, -1.0f, -1.0f), glm::vec3(1.0f, 1.0f, 1.0f));

	bool answer = Geometry::intersect(sphere, bounds);

	print("Intersection of sphere and bounds (11) ", (answer == true));
}

void UnitTests::sphereBoundsIntersectionTest12()
{
	Sphere sphere(glm::vec3(0.0f, 0.0f, 0.0f), 1.0f);
	Bounds bounds(glm::vec3(-1.0f, -1.0f, 1.0f), glm::vec3(1.0f, 1.0f, 1.0f));

	bool answer = Geometry::intersect(sphere, bounds);

	print("Intersection of sphere and bounds (12) ", (answer == true));
}

void UnitTests::sphereBoundsIntersectionTest13()
{
	Sphere sphere(glm::vec3(0.0f, 0.0f, 0.0f), 1.0f);
	Bounds bounds(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.2f, 0.2f, 0.2f));

	bool answer = Geometry::intersect(sphere, bounds);

	print("Intersection of sphere and bounds (13) ", (answer == true));
}

void UnitTests::sphereBoundsIntersectionTest14()
{
	Sphere sphere(glm::vec3(0.0f, 0.0f, 0.0f), 0.1f);
	Bounds bounds(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(1.0f, 1.0f, 1.0f));

	bool answer = Geometry::intersect(sphere, bounds);

	print("Intersection of sphere and bounds (14) ", (answer == true));
}

void UnitTests::sphereBoundsIntersectionTest15()
{
	Sphere sphere(glm::vec3(1.0, 1.0f, 1.0f), 0.5f);
	Bounds bounds(glm::vec3(1.5f, 1.5f, 0.0f), glm::vec3(1.0f, 1.0f, 1.0f));

	bool answer = Geometry::intersect(sphere, bounds);

	print("Intersection of sphere and bounds (15) ", (answer == true));
}

void UnitTests::sphereBoundsIntersectionTest16()
{
	Sphere sphere(glm::vec3(-2.0f, 3.0f, 1.1f), 2.3f);
	Bounds bounds(glm::vec3(5.0f, 10.0f, 4.0f), glm::vec3(1.2f, -2.2f, 4.2f));

	bool answer = Geometry::intersect(sphere, bounds);

	print("Intersection of sphere and bounds (16) ", (answer == false));
}

void UnitTests::sphereBoundsIntersectionTest17()
{
	Sphere sphere(glm::vec3(-2.0f, 0.0f, 0.0f), 0.4f);
	Bounds bounds(glm::vec3(1.0f, -1.0f, 0.0f), glm::vec3(0.2f, 0.2f, 0.2f));

	bool answer = Geometry::intersect(sphere, bounds);

	print("Intersection of sphere and bounds (17) ", (answer == false));
}

void UnitTests::sphereBoundsIntersectionTest18()
{
	Sphere sphere(glm::vec3(-1.0f, -1.0f, -1.0f), 0.4f);
	Bounds bounds(glm::vec3(1.0f, 2.0f, 3.0f), glm::vec3(1.2f, 0.2f, 0.8f));

	bool answer = Geometry::intersect(sphere, bounds);

	print("Intersection of sphere and bounds (18) ", (answer == false));
}

void UnitTests::sphereSphereIntersectionTest0()
{
	Sphere sphere1(glm::vec3(0.0f, 0.0f, 0.0f), 0.75f);
	Sphere sphere2(glm::vec3(1.0f, 0.0f, 0.0f), 0.75f);

	bool answer = Geometry::intersect(sphere1, sphere2);

	print("Intersection of sphere and sphere (0) ", (answer == true));
}

void UnitTests::sphereSphereIntersectionTest1()
{
	Sphere sphere1(glm::vec3(0.0f, 0.0f, 0.0f), 0.75f);
	Sphere sphere2(glm::vec3(0.0f, 1.0f, 0.0f), 0.75f);

	bool answer = Geometry::intersect(sphere1, sphere2);

	print("Intersection of sphere and sphere (1) ", (answer == true));
}

void UnitTests::sphereSphereIntersectionTest2()
{
	Sphere sphere1(glm::vec3(0.0f, 0.0f, 0.0f), 0.75f);
	Sphere sphere2(glm::vec3(0.0f, 0.0f, 1.0f), 0.75f);

	bool answer = Geometry::intersect(sphere1, sphere2);

	print("Intersection of sphere and sphere (2) ", (answer == true));
}

void UnitTests::sphereSphereIntersectionTest3()
{
	Sphere sphere1(glm::vec3(0.0f, 0.0f, 0.0f), 0.75f);
	Sphere sphere2(glm::vec3(-1.0f, 0.0f, 0.0f), 0.75f);

	bool answer = Geometry::intersect(sphere1, sphere2);

	print("Intersection of sphere and sphere (3) ", (answer == true));
}

void UnitTests::sphereSphereIntersectionTest4()
{
	Sphere sphere1(glm::vec3(0.0f, 0.0f, 0.0f), 0.75f);
	Sphere sphere2(glm::vec3(0.0f, -1.0f, 0.0f), 0.75f);

	bool answer = Geometry::intersect(sphere1, sphere2);

	print("Intersection of sphere and sphere (4) ", (answer == true));
}

void UnitTests::sphereSphereIntersectionTest5()
{
	Sphere sphere1(glm::vec3(0.0f, 0.0f, 0.0f), 0.75f);
	Sphere sphere2(glm::vec3(0.0f, 0.0f, -1.0f), 0.75f);

	bool answer = Geometry::intersect(sphere1, sphere2);

	print("Intersection of sphere and sphere (5) ", (answer == true));
}

void UnitTests::sphereSphereIntersectionTest6()
{
	Sphere sphere1(glm::vec3(0.0f, 0.0f, 0.0f), 1.0f);
	Sphere sphere2(glm::vec3(2.0f, 0.0f, 0.0f), 1.0f);

	bool answer = Geometry::intersect(sphere1, sphere2);

	print("Intersection of sphere and sphere (6) ", (answer == true));
}

void UnitTests::sphereSphereIntersectionTest7()
{
	Sphere sphere1(glm::vec3(1.0f, 1.0f, 1.0f), 0.25f);
	Sphere sphere2(glm::vec3(-2.0f, 0.0f, 0.0f), 1.0f);

	bool answer = Geometry::intersect(sphere1, sphere2);

	print("Intersection of sphere and sphere (7) ", (answer == false));
}

void UnitTests::sphereSphereIntersectionTest8()
{
	Sphere sphere1(glm::vec3(1.0f, -2.0f, 3.0f), 0.25f);
	Sphere sphere2(glm::vec3(-20.0f, 1.0f, 5.0f), 0.45f);

	bool answer = Geometry::intersect(sphere1, sphere2);

	print("Intersection of sphere and sphere (8) ", (answer == false));
}

void UnitTests::sphereSphereIntersectionTest9()
{
	Sphere sphere1(glm::vec3(1.0f, -2.0f, 3.0f), 0.5f);
	Sphere sphere2(glm::vec3(1.0f, -2.0f, 3.0f), 0.5f);

	bool answer = Geometry::intersect(sphere1, sphere2);

	print("Intersection of sphere and sphere (9) ", (answer == true));
}

void UnitTests::sphereSphereIntersectionTest10()
{
	Sphere sphere1(glm::vec3(1.0f, -2.0f, 3.0f), 0.5f);
	Sphere sphere2(glm::vec3(1.0f, -2.0f, 3.0f), 0.25f);

	bool answer = Geometry::intersect(sphere1, sphere2);

	print("Intersection of sphere and sphere (10) ", (answer == true));
}

void UnitTests::sphereSphereIntersectionTest11()
{
	Sphere sphere1(glm::vec3(1.0f, -2.0f, 3.0f), 0.25f);
	Sphere sphere2(glm::vec3(1.0f, -2.0f, 3.0f), 0.5f);

	bool answer = Geometry::intersect(sphere1, sphere2);

	print("Intersection of sphere and sphere (11) ", (answer == true));
}

void UnitTests::sphereSphereIntersectionTest12()
{
	Sphere sphere1(glm::vec3(1.0f, -2.0f, 3.0f), 2.5f);
	Sphere sphere2(glm::vec3(1.1f, -1.8f, 3.2f), 0.25f);

	bool answer = Geometry::intersect(sphere1, sphere2);

	print("Intersection of sphere and sphere (12) ", (answer == true));
}

void UnitTests::boundsBoundsIntersectionTest0()
{
	Bounds bounds1(glm::vec3(0.0, 0.0f, 0.0f), glm::vec3(1.0f, 1.0f, 1.0f));
	Bounds bounds2(glm::vec3(1.0f, 0.0f, 0.0f), glm::vec3(1.5f, 1.5f, 1.5f));

	bool answer = Geometry::intersect(bounds1, bounds2);

	print("Intersection of bounds and bounds (0) ", (answer == true));
}

void UnitTests::boundsBoundsIntersectionTest1()
{
	Bounds bounds1(glm::vec3(0.0, 0.0f, 0.0f), glm::vec3(1.0f, 1.0f, 1.0f));
	Bounds bounds2(glm::vec3(-1.0f, 0.0f, 0.0f), glm::vec3(1.5f, 1.5f, 1.5f));

	bool answer = Geometry::intersect(bounds1, bounds2);

	print("Intersection of bounds and bounds (1) ", (answer == true));
}

void UnitTests::boundsBoundsIntersectionTest2()
{
	Bounds bounds1(glm::vec3(0.0, 0.0f, 0.0f), glm::vec3(1.0f, 1.0f, 1.0f));
	Bounds bounds2(glm::vec3(0.0f, 1.0f, 0.0f), glm::vec3(1.5f, 1.5f, 1.5f));

	bool answer = Geometry::intersect(bounds1, bounds2);

	print("Intersection of bounds and bounds (2) ", (answer == true));
}

void UnitTests::boundsBoundsIntersectionTest3()
{
	Bounds bounds1(glm::vec3(0.0, 0.0f, 0.0f), glm::vec3(1.0f, 1.0f, 1.0f));
	Bounds bounds2(glm::vec3(0.0f, -1.0f, 0.0f), glm::vec3(1.5f, 1.5f, 1.5f));

	bool answer = Geometry::intersect(bounds1, bounds2);

	print("Intersection of bounds and bounds (3) ", (answer == true));
}

void UnitTests::boundsBoundsIntersectionTest4()
{
	Bounds bounds1(glm::vec3(0.0, 0.0f, 0.0f), glm::vec3(1.0f, 1.0f, 1.0f));
	Bounds bounds2(glm::vec3(0.0f, 0.0f, 1.0f), glm::vec3(1.5f, 1.5f, 1.5f));

	bool answer = Geometry::intersect(bounds1, bounds2);

	print("Intersection of bounds and bounds (4) ", (answer == true));
}

void UnitTests::boundsBoundsIntersectionTest5()
{
	Bounds bounds1(glm::vec3(0.0, 0.0f, 0.0f), glm::vec3(1.0f, 1.0f, 1.0f));
	Bounds bounds2(glm::vec3(0.0f, 0.0f, -1.0f), glm::vec3(1.5f, 1.5f, 1.5f));

	bool answer = Geometry::intersect(bounds1, bounds2);

	print("Intersection of bounds and bounds (5) ", (answer == true));
}

void UnitTests::boundsBoundsIntersectionTest6()
{
	Bounds bounds1(glm::vec3(0.0, 0.0f, 0.0f), glm::vec3(1.0f, 1.0f, 1.0f));
	Bounds bounds2(glm::vec3(1.0f, 0.0f, 1.0f), glm::vec3(1.0f, 1.0f, 1.0f));

	bool answer = Geometry::intersect(bounds1, bounds2);

	print("Intersection of bounds and bounds (6) ", (answer == true));
}

void UnitTests::boundsBoundsIntersectionTest7()
{
	Bounds bounds1(glm::vec3(0.0, 0.0f, 0.0f), glm::vec3(1.0f, 1.0f, 1.0f));
	Bounds bounds2(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.5f, 0.5f, 0.5f));

	bool answer = Geometry::intersect(bounds1, bounds2);

	print("Intersection of bounds and bounds (7) ", (answer == true));
}

void UnitTests::boundsBoundsIntersectionTest8()
{
	Bounds bounds1(glm::vec3(0.0, 0.0f, 0.0f), glm::vec3(1.0f, 1.0f, 1.0f));
	Bounds bounds2(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.5f, 0.5f, 0.5f));

	bool answer = Geometry::intersect(bounds1, bounds2);

	print("Intersection of bounds and bounds (8) ", (answer == true));
}

void UnitTests::boundsBoundsIntersectionTest9()
{
	Bounds bounds1(glm::vec3(0.0, 0.0f, 0.0f), glm::vec3(2.0f, 1.0f, -1.0f));
	Bounds bounds2(glm::vec3(3.0f, 4.0f, 5.0f), glm::vec3(0.5f, 0.5f, 0.5f));

	bool answer = Geometry::intersect(bounds1, bounds2);

	print("Intersection of bounds and bounds (9) ", (answer == false));
}

void UnitTests::boundsBoundsIntersectionTest10()
{
	Bounds bounds1(glm::vec3(-1.0, 0.5f, -0.7f), glm::vec3(2.0f, 1.0f, -1.0f));
	Bounds bounds2(glm::vec3(3.0f, 4.0f, 5.0f), glm::vec3(0.5f, 0.5f, 0.5f));

	bool answer = Geometry::intersect(bounds1, bounds2);

	print("Intersection of bounds and bounds (10) ", (answer == false));
}
