#ifndef __TEST_POOL_ALLOCATOR_H__
#define __TEST_POOL_ALLOCATOR_H__

#include<core/PoolAllocator.h>
#include<core/Entity.h>
#include<components/Transform.h>

namespace PhysicsEngineTest
{
	template<typename T>
	int poolAllocatorCreateDestroy(int createCount, int destroyCount)
	{
		PhysicsEngine::PoolAllocator<T> allocator;

		for (int i = 0; i < createCount; i++) {
			allocator.construct();
		}

		std::cout << "allocator size after create: " << allocator.getCount() << std::endl;

		for (int i = 0; i < destroyCount; i++) {
			allocator.destruct(0);
		}

		std::cout << "allocator size after destryo: " << allocator.getCount() << std::endl;

		return allocator.getCount();
	}

	TEST(testEngineCore, testPoolAllocator_Entity_10_10)
	{
		EXPECT_EQ(0, poolAllocatorCreateDestroy<PhysicsEngine::Entity>(10, 10));
	}

	TEST(testEngineCore, testPoolAllocator_Entity_20_10)
	{
		EXPECT_EQ(10, poolAllocatorCreateDestroy<PhysicsEngine::Entity>(20, 10));
	}

	TEST(testEngineCore, testPoolAllocator_Entity_100_10)
	{
		EXPECT_EQ(90, poolAllocatorCreateDestroy<PhysicsEngine::Entity>(100, 10));
	}

	TEST(testEngineCore, testPoolAllocator_Entity_200_97)
	{
		EXPECT_EQ(103, poolAllocatorCreateDestroy<PhysicsEngine::Entity>(200, 97));
	}

	TEST(testEngineCore, testPoolAllocator_Entity_200_200)
	{
		EXPECT_EQ(0, poolAllocatorCreateDestroy<PhysicsEngine::Entity>(200, 200));
	}

	TEST(testEngineCore, testPoolAllocator_Entity_201_100)
	{
		EXPECT_EQ(101, poolAllocatorCreateDestroy<PhysicsEngine::Entity>(201, 100));
	}

	TEST(testEngineCore, testPoolAllocator_Entity_201_200)
	{
		EXPECT_EQ(1, poolAllocatorCreateDestroy<PhysicsEngine::Entity>(201, 200));
	}

	TEST(testEngineCore, testPoolAllocator_Entity_2000_1435)
	{
		EXPECT_EQ(565, poolAllocatorCreateDestroy<PhysicsEngine::Entity>(2000, 1435));
	}

	TEST(testEngineCore, testPoolAllocator_Entity_500000_273045)
	{
		EXPECT_EQ(226955, poolAllocatorCreateDestroy<PhysicsEngine::Entity>(500000, 273045));
	}


	// Transforms

	TEST(testEngineCore, testPoolAllocator_Transform_10_10)
	{
		EXPECT_EQ(0, poolAllocatorCreateDestroy<PhysicsEngine::Transform>(10, 10));
	}

	TEST(testEngineCore, testPoolAllocator_Transform_20_10)
	{
		EXPECT_EQ(10, poolAllocatorCreateDestroy<PhysicsEngine::Transform>(20, 10));
	}

	TEST(testEngineCore, testPoolAllocator_Transform_100_10)
	{
		EXPECT_EQ(90, poolAllocatorCreateDestroy<PhysicsEngine::Transform>(100, 10));
	}

	TEST(testEngineCore, testPoolAllocator_Transform_200_97)
	{
		EXPECT_EQ(103, poolAllocatorCreateDestroy<PhysicsEngine::Transform>(200, 97));
	}

	TEST(testEngineCore, testPoolAllocator_Transform_200_200)
	{
		EXPECT_EQ(0, poolAllocatorCreateDestroy<PhysicsEngine::Transform>(200, 200));
	}

	TEST(testEngineCore, testPoolAllocator_Transform_201_100)
	{
		EXPECT_EQ(101, poolAllocatorCreateDestroy<PhysicsEngine::Transform>(201, 100));
	}

	TEST(testEngineCore, testPoolAllocator_Transform_201_200)
	{
		EXPECT_EQ(1, poolAllocatorCreateDestroy<PhysicsEngine::Transform>(201, 200));
	}

	TEST(testEngineCore, testPoolAllocator_Transform_2000_1435)
	{
		EXPECT_EQ(565, poolAllocatorCreateDestroy<PhysicsEngine::Transform>(2000, 1435));
	}

	TEST(testEngineCore, testPoolAllocator_Transform_500000_273045)
	{
		EXPECT_EQ(226955, poolAllocatorCreateDestroy<PhysicsEngine::Transform>(500000, 273045));
	}
}

#endif