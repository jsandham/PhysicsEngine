#include <iostream>

#include <gtest/gtest.h>

#include "../include/TestPoolAllocator.h" 
#include "../include/TestPolynomialRoots.h" 
#include "../include/TestIntersect.h" 

int main()
{
	testing::InitGoogleTest();
	RUN_ALL_TESTS();
}