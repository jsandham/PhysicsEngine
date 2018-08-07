#include <iostream>

#include "glm/glm.hpp"

#include "UnitTests.h"

using namespace PhysicsEngine;

void GeometryUnitTests::run()
{
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

void GeometryUnitTests::print(const std::string testMessage, bool testPassed)
{
	if (testPassed){
		std::cout << "GEOMETRY UNIT TEST: " << testMessage << "PASSED" << std::endl;
	}
	else{
		std::cout << "GEOMETRY UNIT TEST: " << testMessage << "FAILED" << std::endl;
	}
}

void GeometryUnitTests::raySphereIntersectionTest0()
{
	Ray ray(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(1.0f, 0.0f, 0.0f));
	Sphere sphere(glm::vec3(10.0f, 0.0f, 0.0f), 1.0f);

	bool answer = Geometry::intersect(ray, sphere);

	print("Intersection of ray and sphere (0) ", (answer == true));
}

void GeometryUnitTests::raySphereIntersectionTest1()
{
	Ray ray(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
	Sphere sphere(glm::vec3(0.0f, 10.0f, 0.0f), 1.0f);

	bool answer = Geometry::intersect(ray, sphere);

	print("Intersection of ray and sphere (1) ", (answer == true));
}

void GeometryUnitTests::raySphereIntersectionTest2()
{
	Ray ray(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f));
	Sphere sphere(glm::vec3(0.0f, 0.0f, 10.0f), 1.0f);

	bool answer = Geometry::intersect(ray, sphere);

	print("Intersection of ray and sphere (2) ", (answer == true));
}

void GeometryUnitTests::raySphereIntersectionTest3()
{
	Ray ray(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(-1.0f, 0.0f, 0.0f));
	Sphere sphere(glm::vec3(-10.0f, 0.0f, 0.0f), 1.0f);

	bool answer = Geometry::intersect(ray, sphere);

	print("Intersection of ray and sphere (3) ", (answer == true));
}

void GeometryUnitTests::raySphereIntersectionTest4()
{
	Ray ray(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, -1.0f, 0.0f));
	Sphere sphere(glm::vec3(0.0f, -10.0f, 0.0f), 1.0f);

	bool answer = Geometry::intersect(ray, sphere);

	print("Intersection of ray and sphere (4) ", (answer == true));
}

void GeometryUnitTests::raySphereIntersectionTest5()
{
	Ray ray(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, -1.0f));
	Sphere sphere(glm::vec3(0.0f, 0.0f, -10.0f), 1.0f);

	bool answer = Geometry::intersect(ray, sphere);

	print("Intersection of ray and sphere (5) ", (answer == true));
}

void GeometryUnitTests::raySphereIntersectionTest6()
{
	Ray ray(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(1.0f, 1.0f, 1.0f));
	Sphere sphere(glm::vec3(10.0f, 10.0f, 10.0f), 1.0f);

	bool answer = Geometry::intersect(ray, sphere);

	print("Intersection of ray and sphere (6) ", (answer == true));
}

void GeometryUnitTests::raySphereIntersectionTest7()
{
	Ray ray(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(-1.0f, -1.0f, -1.0f));
	Sphere sphere(glm::vec3(-10.0f, -10.0f, -10.0f), 1.0f);

	bool answer = Geometry::intersect(ray, sphere);

	print("Intersection of ray and sphere (7) ", (answer == true));
}

void GeometryUnitTests::raySphereIntersectionTest8()
{
	Ray ray(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(10.0f, 0.5f, 0.5f));
	Sphere sphere(glm::vec3(10.0f, 0.0f, 0.0f), 1.0f);

	bool answer = Geometry::intersect(ray, sphere);

	print("Intersection of ray and sphere (8) ", (answer == true));
}

void GeometryUnitTests::raySphereIntersectionTest9()
{
	Ray ray(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(10.0f, 1.0f, 0.0f));
	Sphere sphere(glm::vec3(10.0f, 0.0f, 0.0f), 1.0f);

	bool answer = Geometry::intersect(ray, sphere);

	print("Intersection of ray and sphere (9) ", (answer == true));
}

void GeometryUnitTests::raySphereIntersectionTest10()
{
	Ray ray(glm::vec3(1.0f, 1.0f, 1.0f), glm::vec3(1.0f, 0.0f, 0.0f));
	Sphere sphere(glm::vec3(10.0f, 0.0f, 0.0f), 2.0f);

	bool answer = Geometry::intersect(ray, sphere);

	print("Intersection of ray and sphere (10) ", (answer == true));
}

void GeometryUnitTests::raySphereIntersectionTest11()
{
	Ray ray(glm::vec3(-1.0f, -1.0f, -1.0f), glm::vec3(1.0f, 0.0f, 0.0f));
	Sphere sphere(glm::vec3(10.0f, 0.0f, 0.0f), 2.0f);

	bool answer = Geometry::intersect(ray, sphere);

	print("Intersection of ray and sphere (11) ", (answer == true));
}

void GeometryUnitTests::raySphereIntersectionTest12()
{
	Ray ray(glm::vec3(1.0f, 2.0f, 3.0f), glm::vec3(-1.0f, -2.0f, -3.0f));
	Sphere sphere(glm::vec3(0.0f, 0.0f, 0.0f), 1.0f);

	bool answer = Geometry::intersect(ray, sphere);

	print("Intersection of ray and sphere (12) ", (answer == true));
}

void GeometryUnitTests::raySphereIntersectionTest13()
{
	Ray ray(glm::vec3(-1.0f, -2.0f, -3.0f), glm::vec3(1.0f, 2.0f, 3.0f));
	Sphere sphere(glm::vec3(0.0f, 0.0f, 0.0f), 1.0f);

	bool answer = Geometry::intersect(ray, sphere);

	print("Intersection of ray and sphere (13) ", (answer == true));
}

void GeometryUnitTests::raySphereIntersectionTest14()
{
	Ray ray(glm::vec3(1.0f, 2.0f, 3.0f), glm::vec3(-1.0f, -2.0f, -3.0f));
	Sphere sphere(glm::vec3(0.2f, 0.3f, 0.1f), 2.0f);

	bool answer = Geometry::intersect(ray, sphere);

	print("Intersection of ray and sphere (14) ", (answer == true));
}

void GeometryUnitTests::raySphereIntersectionTest15()
{
	Ray ray(glm::vec3(-1.0f, -2.0f, -3.0f), glm::vec3(1.0f, 2.0f, 3.0f));
	Sphere sphere(glm::vec3(0.2f, 0.3f, 0.1f), 2.0f);

	bool answer = Geometry::intersect(ray, sphere);

	print("Intersection of ray and sphere (15) ", (answer == true));
}

void GeometryUnitTests::raySphereIntersectionTest16()
{
	Ray ray(glm::vec3(-1.0f, 2.0f, -3.0f), glm::vec3(4.0f, -2.0f, 3.0f));
	Sphere sphere(glm::vec3(-1.0f, 2.0f, -3.0f), 0.5f);

	bool answer = Geometry::intersect(ray, sphere);

	print("Intersection of ray and sphere (16) ", (answer == true));
}

void GeometryUnitTests::raySphereIntersectionTest17()
{
	Ray ray(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(1.0f, 0.0f, 0.0f));
	Sphere sphere(glm::vec3(-10.0f, 0.0f, 0.0f), 1.0f);

	bool answer = Geometry::intersect(ray, sphere);

	print("Intersection of ray and sphere (17) ", (answer == false));
}

void GeometryUnitTests::raySphereIntersectionTest18()
{
	Ray ray(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
	Sphere sphere(glm::vec3(0.0f, -10.0f, 0.0f), 1.0f);

	bool answer = Geometry::intersect(ray, sphere);

	print("Intersection of ray and sphere (18) ", (answer == false));
}

void GeometryUnitTests::raySphereIntersectionTest19()
{
	Ray ray(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f));
	Sphere sphere(glm::vec3(0.0f, 0.0f, -10.0f), 1.0f);

	bool answer = Geometry::intersect(ray, sphere);

	print("Intersection of ray and sphere (19) ", (answer == false));
}

void GeometryUnitTests::raySphereIntersectionTest20()
{
	Ray ray(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(1.0f, 1.0f, 1.0f));
	Sphere sphere(glm::vec3(-10.0f, -10.0f, -10.0f), 1.0f);

	bool answer = Geometry::intersect(ray, sphere);

	print("Intersection of ray and sphere (20) ", (answer == false));
}

void GeometryUnitTests::raySphereIntersectionTest21()
{
	Ray ray(glm::vec3(2.0f, 0.0f, 0.0f), glm::vec3(1.0f, 0.0f, 0.0f));
	Sphere sphere(glm::vec3(1.0f, 0.0f, 0.0f), 0.5f);

	bool answer = Geometry::intersect(ray, sphere);

	print("Intersection of ray and sphere (21) ", (answer == false));
}

void GeometryUnitTests::raySphereIntersectionTest22()
{
	Ray ray(glm::vec3(1.0f, 1.0f, 1.0f), glm::vec3(1.0f, 0.0f, 0.0f));
	Sphere sphere(glm::vec3(10.0f, 0.0f, 0.0f), 1.0f);

	bool answer = Geometry::intersect(ray, sphere);

	print("Intersection of ray and sphere (22) ", (answer == false));
}

void GeometryUnitTests::rayBoundsIntersectionTest0()
{
	Ray ray(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(1.0f, 0.0f, 0.0f));
	Bounds bounds(glm::vec3(10.0f, 0.0f, 0.0f), glm::vec3(1.0f, 1.0f, 1.0f));

	bool answer = Geometry::intersect(ray, bounds);

	print("Intersection of ray and bounds (0) ", (answer == true));
}

void GeometryUnitTests::rayBoundsIntersectionTest1()
{
	Ray ray(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
	Bounds bounds(glm::vec3(0.0f, 10.0f, 0.0f), glm::vec3(1.0f, 1.0f, 1.0f));

	bool answer = Geometry::intersect(ray, bounds);

	print("Intersection of ray and bounds (1) ", (answer == true));
}

void GeometryUnitTests::rayBoundsIntersectionTest2()
{
	Ray ray(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f));
	Bounds bounds(glm::vec3(0.0f, 0.0f, 10.0f), glm::vec3(1.0f, 1.0f, 1.0f));

	bool answer = Geometry::intersect(ray, bounds);

	print("Intersection of ray and bounds (2) ", (answer == true));
}

void GeometryUnitTests::rayBoundsIntersectionTest3()
{
	Ray ray(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(-1.0f, 0.0f, 0.0f));
	Bounds bounds(glm::vec3(-10.0f, 0.0f, 0.0f), glm::vec3(1.0f, 1.0f, 1.0f));

	bool answer = Geometry::intersect(ray, bounds);

	print("Intersection of ray and bounds (3) ", (answer == true));
}

void GeometryUnitTests::rayBoundsIntersectionTest4()
{
	Ray ray(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, -1.0f, 0.0f));
	Bounds bounds(glm::vec3(0.0f, -10.0f, 0.0f), glm::vec3(1.0f, 1.0f, 1.0f));

	bool answer = Geometry::intersect(ray, bounds);

	print("Intersection of ray and bounds (4) ", (answer == true));
}

void GeometryUnitTests::rayBoundsIntersectionTest5()
{
	Ray ray(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, -1.0f));
	Bounds bounds(glm::vec3(0.0f, 0.0f, -10.0f), glm::vec3(1.0f, 1.0f, 1.0f));

	bool answer = Geometry::intersect(ray, bounds);

	print("Intersection of ray and bounds (5) ", (answer == true));
}

void GeometryUnitTests::rayBoundsIntersectionTest6()
{
	Ray ray(glm::vec3(1.0f, 1.0f, 1.0f), glm::vec3(1.0f, 0.0f, 0.0f));
	Bounds bounds(glm::vec3(2.0f, 0.0f, 0.0f), glm::vec3(2.001f, 2.0001f, 2.001f));

	bool answer = Geometry::intersect(ray, bounds);

	print("Intersection of ray and bounds (6) ", (answer == true));
}

void GeometryUnitTests::rayBoundsIntersectionTest7()
{
	Ray ray(glm::vec3(1.0f, 1.0f, 1.0f), glm::vec3(1.0f, 0.0f, 0.0f));
	Bounds bounds(glm::vec3(2.0f, 0.0f, 0.0f), glm::vec3(3.0f, 4.0f, 2.00001f));

	bool answer = Geometry::intersect(ray, bounds);

	print("Intersection of ray and bounds (7) ", (answer == true));
}

void GeometryUnitTests::rayBoundsIntersectionTest8()
{
	Ray ray(glm::vec3(0.0f, 1.0f, 0.0f), glm::vec3(-1.0f, -1.0f, -1.0f));
	Bounds bounds(glm::vec3(0.0f, -2.0f, 0.0f), glm::vec3(4.0f, 4.0f, 4.0f));

	bool answer = Geometry::intersect(ray, bounds);

	print("Intersection of ray and bounds (8) ", (answer == true));
}

void GeometryUnitTests::rayBoundsIntersectionTest9()
{
	Ray ray(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(1.0f, 0.0f, 0.0f));
	Bounds bounds(glm::vec3(-1.0f, 0.0f, 0.0f), glm::vec3(4.0f, 4.0f, 4.0f));

	bool answer = Geometry::intersect(ray, bounds);

	print("Intersection of ray and bounds (9) ", (answer == true));
}

void GeometryUnitTests::rayBoundsIntersectionTest10()
{
	Ray ray(glm::vec3(-2.0f, 0.0f, 0.0f), glm::vec3(1.0f, 0.0f, 0.0f));
	Bounds bounds(glm::vec3(-1.0f, 0.0f, 0.0f), glm::vec3(4.0f, 4.0f, 4.0f));

	bool answer = Geometry::intersect(ray, bounds);

	print("Intersection of ray and bounds (10) ", (answer == true));
}

void GeometryUnitTests::rayBoundsIntersectionTest11()
{
	Ray ray(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(1.0f, 0.0f, 0.0f));
	Bounds bounds(glm::vec3(-1.0f, 0.0f, 0.0f), glm::vec3(1.0f, 1.0f, 1.0f));

	bool answer = Geometry::intersect(ray, bounds);

	print("Intersection of ray and bounds (11) ", (answer == false));
}

void GeometryUnitTests::rayBoundsIntersectionTest12()
{
	Ray ray(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
	Bounds bounds(glm::vec3(0.0f, -1.0f, 0.0f), glm::vec3(1.0f, 1.0f, 1.0f));

	bool answer = Geometry::intersect(ray, bounds);

	print("Intersection of ray and bounds (12) ", (answer == false));
}

void GeometryUnitTests::rayBoundsIntersectionTest13()
{
	Ray ray(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f));
	Bounds bounds(glm::vec3(0.0f, 0.0f, -1.0f), glm::vec3(1.0f, 1.0f, 1.0f));

	bool answer = Geometry::intersect(ray, bounds);

	print("Intersection of ray and bounds (13) ", (answer == false));
}

void GeometryUnitTests::rayBoundsIntersectionTest14()
{
	Ray ray(glm::vec3(1.0f, 1.0f, 1.0f), glm::vec3(0.5f, 0.4f, 1.3f));
	Bounds bounds(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(1.0f, 1.0f, 1.0f));

	bool answer = Geometry::intersect(ray, bounds);

	print("Intersection of ray and bounds (14) ", (answer == false));
}

void GeometryUnitTests::rayBoundsIntersectionTest15()
{
	Ray ray(glm::vec3(1.0f, 1.0f, 1.0f), glm::vec3(-0.5f, -0.4f, -1.3f));
	Bounds bounds(glm::vec3(2.0f, 2.0f, 2.0f), glm::vec3(1.0f, 1.0f, 1.0f));

	bool answer = Geometry::intersect(ray, bounds);

	print("Intersection of ray and bounds (15) ", (answer == false));
}

void GeometryUnitTests::rayBoundsIntersectionTest16()
{
	Ray ray(glm::vec3(1.0f, 1.0f, 1.0f), glm::vec3(1.0f, 0.0f, 0.0f));
	Bounds bounds(glm::vec3(2.0f, 0.0f, 0.0f), glm::vec3(2.0f, 2.0f, 2.0f));

	bool answer = Geometry::intersect(ray, bounds);

	print("Intersection of ray and bounds (16) ", (answer == true));
}

void GeometryUnitTests::rayBoundsIntersectionTest17()
{
	Ray ray(glm::vec3(1.0f, 1.0f, 1.0f), glm::vec3(1.0f, 0.0f, 0.0f));
	Bounds bounds(glm::vec3(2.0f, 0.0f, 0.0f), glm::vec3(3.0f, 4.0f, 2.0f));

	bool answer = Geometry::intersect(ray, bounds);

	print("Intersection of ray and bounds (17) ", (answer == true));
}

void GeometryUnitTests::rayBoundsIntersectionTest18()
{
	Ray ray(glm::vec3(-1.0f, 10.0f, 2.2f), glm::vec3(1.0f, 0.1f, -2.3f));
	Bounds bounds(glm::vec3(-1.015f, 9.85f, 1.9f), glm::vec3(2.8f, 1.234f, 2.1f));

	bool answer = Geometry::intersect(ray, bounds);

	print("Intersection of ray and bounds (18) ", (answer == true));
}


void GeometryUnitTests::sphereBoundsIntersectionTest0()
{
	Sphere sphere(glm::vec3(0.0f, 0.0f, 0.0f), 1.0f);
	Bounds bounds(glm::vec3(1.0f, 0.0f, 0.0f), glm::vec3(1.0f, 1.0f, 1.0f));

	bool answer = Geometry::intersect(sphere, bounds);

	print("Intersection of sphere and bounds (0) ", (answer == true));
}

void GeometryUnitTests::sphereBoundsIntersectionTest1()
{
	Sphere sphere(glm::vec3(0.0f, 0.0f, 0.0f), 1.0f);
	Bounds bounds(glm::vec3(1.0f, 0.0f, 0.0f), glm::vec3(1.0f, 1.0f, 1.0f));

	bool answer = Geometry::intersect(sphere, bounds);

	print("Intersection of sphere and bounds (1) ", (answer == true));
}

void GeometryUnitTests::sphereBoundsIntersectionTest2()
{
	Sphere sphere(glm::vec3(0.0f, 0.0f, 0.0f), 1.0f);
	Bounds bounds(glm::vec3(-1.0f, 0.0f, 0.0f), glm::vec3(1.0f, 1.0f, 1.0f));

	bool answer = Geometry::intersect(sphere, bounds);

	print("Intersection of sphere and bounds (2) ", (answer == true));
}

void GeometryUnitTests::sphereBoundsIntersectionTest3()
{
	Sphere sphere(glm::vec3(0.0f, 0.0f, 0.0f), 1.0f);
	Bounds bounds(glm::vec3(0.0f, 1.0f, 0.0f), glm::vec3(1.0f, 1.0f, 1.0f));

	bool answer = Geometry::intersect(sphere, bounds);

	print("Intersection of sphere and bounds (3) ", (answer == true));
}

void GeometryUnitTests::sphereBoundsIntersectionTest4()
{
	Sphere sphere(glm::vec3(0.0f, 0.0f, 0.0f), 1.0f);
	Bounds bounds(glm::vec3(0.0f, -1.0f, 0.0f), glm::vec3(1.0f, 1.0f, 1.0f));

	bool answer = Geometry::intersect(sphere, bounds);

	print("Intersection of sphere and bounds (4) ", (answer == true));
}

void GeometryUnitTests::sphereBoundsIntersectionTest5()
{
	Sphere sphere(glm::vec3(0.0f, 0.0f, 0.0f), 1.0f);
	Bounds bounds(glm::vec3(0.0f, 0.0f, 1.0f), glm::vec3(1.0f, 1.0f, 1.0f));

	bool answer = Geometry::intersect(sphere, bounds);

	print("Intersection of sphere and bounds (5) ", (answer == true));
}

void GeometryUnitTests::sphereBoundsIntersectionTest6()
{
	Sphere sphere(glm::vec3(0.0f, 0.0f, 0.0f), 1.0f);
	Bounds bounds(glm::vec3(0.0f, 0.0f, -1.0f), glm::vec3(1.0f, 1.0f, 1.0f));

	bool answer = Geometry::intersect(sphere, bounds);

	print("Intersection of sphere and bounds (6) ", (answer == true));
}

void GeometryUnitTests::sphereBoundsIntersectionTest7()
{
	Sphere sphere(glm::vec3(0.0f, 0.0f, 0.0f), 1.0f);
	Bounds bounds(glm::vec3(1.0f, 1.0f, 1.0f), glm::vec3(1.0f, 1.0f, 1.0f));

	bool answer = Geometry::intersect(sphere, bounds);

	print("Intersection of sphere and bounds (7) ", (answer == true));
}

void GeometryUnitTests::sphereBoundsIntersectionTest8()
{
	Sphere sphere(glm::vec3(0.0f, 0.0f, 0.0f), 1.0f);
	Bounds bounds(glm::vec3(-1.0f, -1.0f, -1.0f), glm::vec3(1.0f, 1.0f, 1.0f));

	bool answer = Geometry::intersect(sphere, bounds);

	print("Intersection of sphere and bounds (8) ", (answer == true));
}

void GeometryUnitTests::sphereBoundsIntersectionTest9()
{
	Sphere sphere(glm::vec3(0.0f, 0.0f, 0.0f), 1.0f);
	Bounds bounds(glm::vec3(-1.0f, 1.0f, -1.0f), glm::vec3(1.0f, 1.0f, 1.0f));

	bool answer = Geometry::intersect(sphere, bounds);

	print("Intersection of sphere and bounds (9) ", (answer == true));
}

void GeometryUnitTests::sphereBoundsIntersectionTest10()
{
	Sphere sphere(glm::vec3(0.0f, 0.0f, 0.0f), 1.0f);
	Bounds bounds(glm::vec3(1.0f, -1.0f, 1.0f), glm::vec3(1.0f, 1.0f, 1.0f));

	bool answer = Geometry::intersect(sphere, bounds);

	print("Intersection of sphere and bounds (10) ", (answer == true));
}

void GeometryUnitTests::sphereBoundsIntersectionTest11()
{
	Sphere sphere(glm::vec3(0.0f, 0.0f, 0.0f), 1.0f);
	Bounds bounds(glm::vec3(1.0f, -1.0f, -1.0f), glm::vec3(1.0f, 1.0f, 1.0f));

	bool answer = Geometry::intersect(sphere, bounds);

	print("Intersection of sphere and bounds (11) ", (answer == true));
}

void GeometryUnitTests::sphereBoundsIntersectionTest12()
{
	Sphere sphere(glm::vec3(0.0f, 0.0f, 0.0f), 1.0f);
	Bounds bounds(glm::vec3(-1.0f, -1.0f, 1.0f), glm::vec3(1.0f, 1.0f, 1.0f));

	bool answer = Geometry::intersect(sphere, bounds);

	print("Intersection of sphere and bounds (12) ", (answer == true));
}

void GeometryUnitTests::sphereBoundsIntersectionTest13()
{
	Sphere sphere(glm::vec3(0.0f, 0.0f, 0.0f), 1.0f);
	Bounds bounds(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.2f, 0.2f, 0.2f));

	bool answer = Geometry::intersect(sphere, bounds);

	print("Intersection of sphere and bounds (13) ", (answer == true));
}

void GeometryUnitTests::sphereBoundsIntersectionTest14()
{
	Sphere sphere(glm::vec3(0.0f, 0.0f, 0.0f), 0.1f);
	Bounds bounds(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(1.0f, 1.0f, 1.0f));

	bool answer = Geometry::intersect(sphere, bounds);

	print("Intersection of sphere and bounds (14) ", (answer == true));
}

void GeometryUnitTests::sphereBoundsIntersectionTest15()
{
	Sphere sphere(glm::vec3(1.0, 1.0f, 1.0f), 0.5f);
	Bounds bounds(glm::vec3(1.5f, 1.5f, 0.0f), glm::vec3(1.0f, 1.0f, 1.0f));

	bool answer = Geometry::intersect(sphere, bounds);

	print("Intersection of sphere and bounds (15) ", (answer == true));
}

void GeometryUnitTests::sphereBoundsIntersectionTest16()
{
	Sphere sphere(glm::vec3(-2.0f, 3.0f, 1.1f), 2.3f);
	Bounds bounds(glm::vec3(5.0f, 10.0f, 4.0f), glm::vec3(1.2f, -2.2f, 4.2f));

	bool answer = Geometry::intersect(sphere, bounds);

	print("Intersection of sphere and bounds (16) ", (answer == false));
}

void GeometryUnitTests::sphereBoundsIntersectionTest17()
{
	Sphere sphere(glm::vec3(-2.0f, 0.0f, 0.0f), 0.4f);
	Bounds bounds(glm::vec3(1.0f, -1.0f, 0.0f), glm::vec3(0.2f, 0.2f, 0.2f));

	bool answer = Geometry::intersect(sphere, bounds);

	print("Intersection of sphere and bounds (17) ", (answer == false));
}

void GeometryUnitTests::sphereBoundsIntersectionTest18()
{
	Sphere sphere(glm::vec3(-1.0f, -1.0f, -1.0f), 0.4f);
	Bounds bounds(glm::vec3(1.0f, 2.0f, 3.0f), glm::vec3(1.2f, 0.2f, 0.8f));

	bool answer = Geometry::intersect(sphere, bounds);

	print("Intersection of sphere and bounds (18) ", (answer == false));
}

void GeometryUnitTests::sphereSphereIntersectionTest0()
{
	Sphere sphere1(glm::vec3(0.0f, 0.0f, 0.0f), 0.75f);
	Sphere sphere2(glm::vec3(1.0f, 0.0f, 0.0f), 0.75f);

	bool answer = Geometry::intersect(sphere1, sphere2);

	print("Intersection of sphere and sphere (0) ", (answer == true));
}

void GeometryUnitTests::sphereSphereIntersectionTest1()
{
	Sphere sphere1(glm::vec3(0.0f, 0.0f, 0.0f), 0.75f);
	Sphere sphere2(glm::vec3(0.0f, 1.0f, 0.0f), 0.75f);

	bool answer = Geometry::intersect(sphere1, sphere2);

	print("Intersection of sphere and sphere (1) ", (answer == true));
}

void GeometryUnitTests::sphereSphereIntersectionTest2()
{
	Sphere sphere1(glm::vec3(0.0f, 0.0f, 0.0f), 0.75f);
	Sphere sphere2(glm::vec3(0.0f, 0.0f, 1.0f), 0.75f);

	bool answer = Geometry::intersect(sphere1, sphere2);

	print("Intersection of sphere and sphere (2) ", (answer == true));
}

void GeometryUnitTests::sphereSphereIntersectionTest3()
{
	Sphere sphere1(glm::vec3(0.0f, 0.0f, 0.0f), 0.75f);
	Sphere sphere2(glm::vec3(-1.0f, 0.0f, 0.0f), 0.75f);

	bool answer = Geometry::intersect(sphere1, sphere2);

	print("Intersection of sphere and sphere (3) ", (answer == true));
}

void GeometryUnitTests::sphereSphereIntersectionTest4()
{
	Sphere sphere1(glm::vec3(0.0f, 0.0f, 0.0f), 0.75f);
	Sphere sphere2(glm::vec3(0.0f, -1.0f, 0.0f), 0.75f);

	bool answer = Geometry::intersect(sphere1, sphere2);

	print("Intersection of sphere and sphere (4) ", (answer == true));
}

void GeometryUnitTests::sphereSphereIntersectionTest5()
{
	Sphere sphere1(glm::vec3(0.0f, 0.0f, 0.0f), 0.75f);
	Sphere sphere2(glm::vec3(0.0f, 0.0f, -1.0f), 0.75f);

	bool answer = Geometry::intersect(sphere1, sphere2);

	print("Intersection of sphere and sphere (5) ", (answer == true));
}

void GeometryUnitTests::sphereSphereIntersectionTest6()
{
	Sphere sphere1(glm::vec3(0.0f, 0.0f, 0.0f), 1.0f);
	Sphere sphere2(glm::vec3(2.0f, 0.0f, 0.0f), 1.0f);

	bool answer = Geometry::intersect(sphere1, sphere2);

	print("Intersection of sphere and sphere (6) ", (answer == true));
}

void GeometryUnitTests::sphereSphereIntersectionTest7()
{
	Sphere sphere1(glm::vec3(1.0f, 1.0f, 1.0f), 0.25f);
	Sphere sphere2(glm::vec3(-2.0f, 0.0f, 0.0f), 1.0f);

	bool answer = Geometry::intersect(sphere1, sphere2);

	print("Intersection of sphere and sphere (7) ", (answer == false));
}

void GeometryUnitTests::sphereSphereIntersectionTest8()
{
	Sphere sphere1(glm::vec3(1.0f, -2.0f, 3.0f), 0.25f);
	Sphere sphere2(glm::vec3(-20.0f, 1.0f, 5.0f), 0.45f);

	bool answer = Geometry::intersect(sphere1, sphere2);

	print("Intersection of sphere and sphere (8) ", (answer == false));
}

void GeometryUnitTests::sphereSphereIntersectionTest9()
{
	Sphere sphere1(glm::vec3(1.0f, -2.0f, 3.0f), 0.5f);
	Sphere sphere2(glm::vec3(1.0f, -2.0f, 3.0f), 0.5f);

	bool answer = Geometry::intersect(sphere1, sphere2);

	print("Intersection of sphere and sphere (9) ", (answer == true));
}

void GeometryUnitTests::sphereSphereIntersectionTest10()
{
	Sphere sphere1(glm::vec3(1.0f, -2.0f, 3.0f), 0.5f);
	Sphere sphere2(glm::vec3(1.0f, -2.0f, 3.0f), 0.25f);

	bool answer = Geometry::intersect(sphere1, sphere2);

	print("Intersection of sphere and sphere (10) ", (answer == true));
}

void GeometryUnitTests::sphereSphereIntersectionTest11()
{
	Sphere sphere1(glm::vec3(1.0f, -2.0f, 3.0f), 0.25f);
	Sphere sphere2(glm::vec3(1.0f, -2.0f, 3.0f), 0.5f);

	bool answer = Geometry::intersect(sphere1, sphere2);

	print("Intersection of sphere and sphere (11) ", (answer == true));
}

void GeometryUnitTests::sphereSphereIntersectionTest12()
{
	Sphere sphere1(glm::vec3(1.0f, -2.0f, 3.0f), 2.5f);
	Sphere sphere2(glm::vec3(1.1f, -1.8f, 3.2f), 0.25f);

	bool answer = Geometry::intersect(sphere1, sphere2);

	print("Intersection of sphere and sphere (12) ", (answer == true));
}

void GeometryUnitTests::boundsBoundsIntersectionTest0()
{
	Bounds bounds1(glm::vec3(0.0, 0.0f, 0.0f), glm::vec3(1.0f, 1.0f, 1.0f));
	Bounds bounds2(glm::vec3(1.0f, 0.0f, 0.0f), glm::vec3(1.5f, 1.5f, 1.5f));

	bool answer = Geometry::intersect(bounds1, bounds2);

	print("Intersection of bounds and bounds (0) ", (answer == true));
}

void GeometryUnitTests::boundsBoundsIntersectionTest1()
{
	Bounds bounds1(glm::vec3(0.0, 0.0f, 0.0f), glm::vec3(1.0f, 1.0f, 1.0f));
	Bounds bounds2(glm::vec3(-1.0f, 0.0f, 0.0f), glm::vec3(1.5f, 1.5f, 1.5f));

	bool answer = Geometry::intersect(bounds1, bounds2);

	print("Intersection of bounds and bounds (1) ", (answer == true));
}

void GeometryUnitTests::boundsBoundsIntersectionTest2()
{
	Bounds bounds1(glm::vec3(0.0, 0.0f, 0.0f), glm::vec3(1.0f, 1.0f, 1.0f));
	Bounds bounds2(glm::vec3(0.0f, 1.0f, 0.0f), glm::vec3(1.5f, 1.5f, 1.5f));

	bool answer = Geometry::intersect(bounds1, bounds2);

	print("Intersection of bounds and bounds (2) ", (answer == true));
}

void GeometryUnitTests::boundsBoundsIntersectionTest3()
{
	Bounds bounds1(glm::vec3(0.0, 0.0f, 0.0f), glm::vec3(1.0f, 1.0f, 1.0f));
	Bounds bounds2(glm::vec3(0.0f, -1.0f, 0.0f), glm::vec3(1.5f, 1.5f, 1.5f));

	bool answer = Geometry::intersect(bounds1, bounds2);

	print("Intersection of bounds and bounds (3) ", (answer == true));
}

void GeometryUnitTests::boundsBoundsIntersectionTest4()
{
	Bounds bounds1(glm::vec3(0.0, 0.0f, 0.0f), glm::vec3(1.0f, 1.0f, 1.0f));
	Bounds bounds2(glm::vec3(0.0f, 0.0f, 1.0f), glm::vec3(1.5f, 1.5f, 1.5f));

	bool answer = Geometry::intersect(bounds1, bounds2);

	print("Intersection of bounds and bounds (4) ", (answer == true));
}

void GeometryUnitTests::boundsBoundsIntersectionTest5()
{
	Bounds bounds1(glm::vec3(0.0, 0.0f, 0.0f), glm::vec3(1.0f, 1.0f, 1.0f));
	Bounds bounds2(glm::vec3(0.0f, 0.0f, -1.0f), glm::vec3(1.5f, 1.5f, 1.5f));

	bool answer = Geometry::intersect(bounds1, bounds2);

	print("Intersection of bounds and bounds (5) ", (answer == true));
}

void GeometryUnitTests::boundsBoundsIntersectionTest6()
{
	Bounds bounds1(glm::vec3(0.0, 0.0f, 0.0f), glm::vec3(1.0f, 1.0f, 1.0f));
	Bounds bounds2(glm::vec3(1.0f, 0.0f, 1.0f), glm::vec3(1.0f, 1.0f, 1.0f));

	bool answer = Geometry::intersect(bounds1, bounds2);

	print("Intersection of bounds and bounds (6) ", (answer == true));
}

void GeometryUnitTests::boundsBoundsIntersectionTest7()
{
	Bounds bounds1(glm::vec3(0.0, 0.0f, 0.0f), glm::vec3(1.0f, 1.0f, 1.0f));
	Bounds bounds2(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.5f, 0.5f, 0.5f));

	bool answer = Geometry::intersect(bounds1, bounds2);

	print("Intersection of bounds and bounds (7) ", (answer == true));
}

void GeometryUnitTests::boundsBoundsIntersectionTest8()
{
	Bounds bounds1(glm::vec3(0.0, 0.0f, 0.0f), glm::vec3(1.0f, 1.0f, 1.0f));
	Bounds bounds2(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.5f, 0.5f, 0.5f));

	bool answer = Geometry::intersect(bounds1, bounds2);

	print("Intersection of bounds and bounds (8) ", (answer == true));
}

void GeometryUnitTests::boundsBoundsIntersectionTest9()
{
	Bounds bounds1(glm::vec3(0.0, 0.0f, 0.0f), glm::vec3(2.0f, 1.0f, -1.0f));
	Bounds bounds2(glm::vec3(3.0f, 4.0f, 5.0f), glm::vec3(0.5f, 0.5f, 0.5f));

	bool answer = Geometry::intersect(bounds1, bounds2);

	print("Intersection of bounds and bounds (9) ", (answer == false));
}

void GeometryUnitTests::boundsBoundsIntersectionTest10()
{
	Bounds bounds1(glm::vec3(-1.0, 0.5f, -0.7f), glm::vec3(2.0f, 1.0f, -1.0f));
	Bounds bounds2(glm::vec3(3.0f, 4.0f, 5.0f), glm::vec3(0.5f, 0.5f, 0.5f));

	bool answer = Geometry::intersect(bounds1, bounds2);

	print("Intersection of bounds and bounds (10) ", (answer == false));
}
