#ifndef __COMMAND_MANAGER_H__
#define __COMMAND_MANAGER_H__

#include <vector>
#include <queue>

#include "Command.h"

#include "core/Input.h"

namespace PhysicsEditor
{
	class CommandManager
	{
		private:
			static int counter;
			static std::vector<Command*> commandHistory;
			static std::queue<Command*> commandQueue;

		public:
			CommandManager();
			~CommandManager();

			void update(PhysicsEngine::Input input);

			static void addCommand(Command* command);
			static void executeCommand();
			static void undoCommand();
			static bool canUndo();
			static bool canRedo();
	};
}

#endif
