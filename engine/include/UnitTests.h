#ifndef __UNIT_TESTS_H__
#define __UNIT_TESTS_H__

#include <string>

namespace PhysicsEngine
{
	class UnitTests
	{
		private:
			static void print(const std::string textMessage, bool testPassed);

			// Core tests

			static void poolAllocatorTest0();
			static void poolAllocatorTest1();
			static void poolAllocatorTest2();
			static void poolAllocatorTest3();
			static void poolAllocatorTest4();
			static void poolAllocatorTest5();

			// Geometry tests

			static void rayPlaneIntersectionTest0();
			static void rayPlaneIntersectionTest1();
			static void rayPlaneIntersectionTest2();
			static void rayPlaneIntersectionTest3();
			static void rayPlaneIntersectionTest4();
			static void rayPlaneIntersectionTest5();
			static void rayPlaneIntersectionTest6();
			static void rayPlaneIntersectionTest7();
			static void rayPlaneIntersectionTest8();
			static void rayPlaneIntersectionTest9();

			static void raySphereIntersectionTest0();
			static void raySphereIntersectionTest1();
			static void raySphereIntersectionTest2();
			static void raySphereIntersectionTest3();
			static void raySphereIntersectionTest4();
			static void raySphereIntersectionTest5();
			static void raySphereIntersectionTest6();
			static void raySphereIntersectionTest7();
			static void raySphereIntersectionTest8();
			static void raySphereIntersectionTest9();
			static void raySphereIntersectionTest10();
			static void raySphereIntersectionTest11();
			static void raySphereIntersectionTest12();
			static void raySphereIntersectionTest13();
			static void raySphereIntersectionTest14();
			static void raySphereIntersectionTest15();
			static void raySphereIntersectionTest16();
			static void raySphereIntersectionTest17();
			static void raySphereIntersectionTest18();
			static void raySphereIntersectionTest19();
			static void raySphereIntersectionTest20();
			static void raySphereIntersectionTest21();
			static void raySphereIntersectionTest22();

			static void rayAABBIntersectionTest0();
			static void rayAABBIntersectionTest1();
			static void rayAABBIntersectionTest2();
			static void rayAABBIntersectionTest3();
			static void rayAABBIntersectionTest4();
			static void rayAABBIntersectionTest5();
			static void rayAABBIntersectionTest6();
			static void rayAABBIntersectionTest7();
			static void rayAABBIntersectionTest8();
			static void rayAABBIntersectionTest9();
			static void rayAABBIntersectionTest10();
			static void rayAABBIntersectionTest11();
			static void rayAABBIntersectionTest12();
			static void rayAABBIntersectionTest13();
			static void rayAABBIntersectionTest14();
			static void rayAABBIntersectionTest15();
			static void rayAABBIntersectionTest16();
			static void rayAABBIntersectionTest17();
			static void rayAABBIntersectionTest18();

			static void sphereAABBIntersectionTest0();
			static void sphereAABBIntersectionTest1();
			static void sphereAABBIntersectionTest2();
			static void sphereAABBIntersectionTest3();
			static void sphereAABBIntersectionTest4();
			static void sphereAABBIntersectionTest5();
			static void sphereAABBIntersectionTest6();
			static void sphereAABBIntersectionTest7();
			static void sphereAABBIntersectionTest8();
			static void sphereAABBIntersectionTest9();
			static void sphereAABBIntersectionTest10();
			static void sphereAABBIntersectionTest11();
			static void sphereAABBIntersectionTest12();
			static void sphereAABBIntersectionTest13();
			static void sphereAABBIntersectionTest14();
			static void sphereAABBIntersectionTest15();
			static void sphereAABBIntersectionTest16();
			static void sphereAABBIntersectionTest17();
			static void sphereAABBIntersectionTest18();

			static void sphereSphereIntersectionTest0();
			static void sphereSphereIntersectionTest1();
			static void sphereSphereIntersectionTest2();
			static void sphereSphereIntersectionTest3();
			static void sphereSphereIntersectionTest4();
			static void sphereSphereIntersectionTest5();
			static void sphereSphereIntersectionTest6();
			static void sphereSphereIntersectionTest7();
			static void sphereSphereIntersectionTest8();
			static void sphereSphereIntersectionTest9();
			static void sphereSphereIntersectionTest10();
			static void sphereSphereIntersectionTest11();
			static void sphereSphereIntersectionTest12();

			static void sphereFrustumIntersectionTest0();
			static void sphereFrustumIntersectionTest1();
			static void sphereFrustumIntersectionTest2();
			static void sphereFrustumIntersectionTest3();
			static void sphereFrustumIntersectionTest4();
			static void sphereFrustumIntersectionTest5();
			static void sphereFrustumIntersectionTest6();
			static void sphereFrustumIntersectionTest7();
			static void sphereFrustumIntersectionTest8();
			static void sphereFrustumIntersectionTest9();
			static void sphereFrustumIntersectionTest10();
			static void sphereFrustumIntersectionTest11();
			static void sphereFrustumIntersectionTest12();
			static void sphereFrustumIntersectionTest13();

			static void aabbAABBIntersectionTest0();
			static void aabbAABBIntersectionTest1();
			static void aabbAABBIntersectionTest2();
			static void aabbAABBIntersectionTest3();
			static void aabbAABBIntersectionTest4();
			static void aabbAABBIntersectionTest5();
			static void aabbAABBIntersectionTest6();
			static void aabbAABBIntersectionTest7();
			static void aabbAABBIntersectionTest8();
			static void aabbAABBIntersectionTest9();
			static void aabbAABBIntersectionTest10();

		public:
			static void run();
	};
}

#endif