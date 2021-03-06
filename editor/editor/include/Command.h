#ifndef __COMMAND_H__
#define __COMMAND_H__

#include <vector>

#include "core/World.h"
#include "core/Guid.h"

namespace PhysicsEditor
{
class Command
{
  public:
    Command();
    virtual ~Command() = 0;

    virtual void execute() = 0;
    virtual void undo() = 0;
};

class RecordEntityCreationCommand : public Command
{
private:
	PhysicsEngine::World* mWorld;
	PhysicsEngine::Guid mEntityId;
	std::vector<char> mData;

	bool mRedo;

public:
	RecordEntityCreationCommand(PhysicsEngine::World * world, PhysicsEngine::Guid entityId, std::vector<char> data)
	{
		mWorld = world;
		mEntityId = entityId;
		mData = data;

		mRedo = false;
	}

	void execute()
	{
		if (mRedo)
		{
			//mWorld->createEntity(mData);
		}
	}

	void undo()
	{
		mWorld->latentDestroyEntity(mEntityId);

		mRedo = true;
	}
};

//template<typename T>
//class RecordComponentAdditionCommand : public Command
//{
//private:
//	PhysicsEngine::World* mWorld;
//	PhysicsEngine::Guid mComponentId;
//	std::vector<char> mData;
//
//	bool mRedo;
//
//public:
//	RecordComponentAdditionCommand(PhysicsEngine::World* world, PhysicsEngine::Guid componentId, std::vector<char> data)
//	{
//		mWorld = world;
//		mComponentId = componentId;
//		mData = data;
//
//		mRedo = false;
//	}
//
//	void execute()
//	{
//		if (mRedo)
//		{
//			mEntity->addComponent<T>(mWorld);
//			//mWorld->createEntity(mData);
//		}
//	}
//
//	void undo()
//	{
//		//mWorld->latentDestroyEntity(mEntityId);
//
//		mRedo = true;
//	}
//};


template<typename T>
class RecordObjectCommand : public Command
{
private:
	T* mT;
	std::vector<char> mOld;
	std::vector<char> mNew;

public:
	RecordObjectCommand(T* t, std::vector<char> oldData, std::vector<char> newData)
	{
		mT = t;
		mOld = oldData;
		mNew = newData;
	}

	void execute()
	{
		mT->deserialize(mNew);
	}

	void undo()
	{
		mT->deserialize(mOld);
	}
};
} // namespace PhysicsEditor

#endif
