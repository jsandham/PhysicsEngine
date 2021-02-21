#ifndef __UNDO_H__
#define __UNDO_H__

#include <vector>
#include <queue>

#include "core/Guid.h"
#include "core/Entity.h"
#include "components/Component.h"

#include "Command.h"
#include "EditorClipboard.h"

namespace PhysicsEditor
{
	class Undo
	{
	private:
		static std::vector<std::pair<PhysicsEngine::Entity*, std::vector<char> >> mCreatedEntityRecords;
		static std::vector<std::pair<PhysicsEngine::Entity*, std::vector<char> >> mEntityRecords;
		static std::vector<std::pair<PhysicsEngine::Component*, std::vector<char> >> mAddComponentRecords;
		static std::vector<std::pair<PhysicsEngine::Component*, std::vector<char> >> mComponentRecords;

		static int counter;
		static std::vector<Command*> commandHistory;
		static std::queue<Command*> commandQueue;

	public:
		Undo() = delete;
		~Undo() = delete;

		static void updateUndoStack(EditorClipboard& clipboard);
		static void clearUndoStack();

		static void recordEntityCreation(PhysicsEngine::Entity* entity);
		static void recordEntity(PhysicsEngine::Entity* entity);
		static void recordComponent(PhysicsEngine::Component* component);

		template<typename T>
		static T* addComponent(PhysicsEngine::Entity* entity)
		{
			//T* component = entity->addComponent<T>();
			//Undo::mAddComponentRecords.push_back(std::make_pair(component, component->serialize()));
			return nullptr;
		}

		static void addCommand(Command* command);
		static void executeCommand();
		static void undoCommand();
		static bool canUndo();
		static bool canRedo();
	};
}

#endif