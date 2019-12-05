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

			// Geometry tests

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

			static void rayBoundsIntersectionTest0();
			static void rayBoundsIntersectionTest1();
			static void rayBoundsIntersectionTest2();
			static void rayBoundsIntersectionTest3();
			static void rayBoundsIntersectionTest4();
			static void rayBoundsIntersectionTest5();
			static void rayBoundsIntersectionTest6();
			static void rayBoundsIntersectionTest7();
			static void rayBoundsIntersectionTest8();
			static void rayBoundsIntersectionTest9();
			static void rayBoundsIntersectionTest10();
			static void rayBoundsIntersectionTest11();
			static void rayBoundsIntersectionTest12();
			static void rayBoundsIntersectionTest13();
			static void rayBoundsIntersectionTest14();
			static void rayBoundsIntersectionTest15();
			static void rayBoundsIntersectionTest16();
			static void rayBoundsIntersectionTest17();
			static void rayBoundsIntersectionTest18();

			static void sphereBoundsIntersectionTest0();
			static void sphereBoundsIntersectionTest1();
			static void sphereBoundsIntersectionTest2();
			static void sphereBoundsIntersectionTest3();
			static void sphereBoundsIntersectionTest4();
			static void sphereBoundsIntersectionTest5();
			static void sphereBoundsIntersectionTest6();
			static void sphereBoundsIntersectionTest7();
			static void sphereBoundsIntersectionTest8();
			static void sphereBoundsIntersectionTest9();
			static void sphereBoundsIntersectionTest10();
			static void sphereBoundsIntersectionTest11();
			static void sphereBoundsIntersectionTest12();
			static void sphereBoundsIntersectionTest13();
			static void sphereBoundsIntersectionTest14();
			static void sphereBoundsIntersectionTest15();
			static void sphereBoundsIntersectionTest16();
			static void sphereBoundsIntersectionTest17();
			static void sphereBoundsIntersectionTest18();

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

			static void boundsBoundsIntersectionTest0();
			static void boundsBoundsIntersectionTest1();
			static void boundsBoundsIntersectionTest2();
			static void boundsBoundsIntersectionTest3();
			static void boundsBoundsIntersectionTest4();
			static void boundsBoundsIntersectionTest5();
			static void boundsBoundsIntersectionTest6();
			static void boundsBoundsIntersectionTest7();
			static void boundsBoundsIntersectionTest8();
			static void boundsBoundsIntersectionTest9();
			static void boundsBoundsIntersectionTest10();

		public:
			static void run();
	};
}

#endif