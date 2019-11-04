#include "../include/CommandManager.h"

#include "core/Log.h"

using namespace PhysicsEditor;

int CommandManager::counter = 0;
std::vector<Command*> CommandManager::commandHistory;
std::queue<Command*> CommandManager::commandQueue;

CommandManager::CommandManager()
{

}

CommandManager::~CommandManager()
{
	// delete any remaining commands in queue and history
	for (size_t i = 0; i < commandHistory.size(); i++) {
		delete commandHistory[i];
	}
	
	while (!commandQueue.empty()) {
		Command* command = commandQueue.front();
		commandQueue.pop();

		delete command;
	}
}

void CommandManager::update(PhysicsEngine::Input input)
{
	while (!commandQueue.empty()) {
		Command* command = commandQueue.front();
		commandQueue.pop();

		command->execute();

		commandHistory.push_back(command);
		counter++;
	}

	//bool ctrlDown = PhysicsEngine::getKey(input, PhysicsEngine::KeyCode::LCtrl);
	//bool ZDown = PhysicsEngine::getKeyDown(input, PhysicsEngine::KeyCode::Z);
	//std::string message = "ctrl down: " + std::to_string(ctrlDown) + " z down: " + std::to_string(ZDown) + "\n";
	//PhysicsEngine::Log::info(&message[0]);

	if (PhysicsEngine::getKey(input, PhysicsEngine::KeyCode::LCtrl) && PhysicsEngine::getKeyDown(input, PhysicsEngine::KeyCode::Z)) {
		undoCommand();
	}

	if (PhysicsEngine::getKey(input, PhysicsEngine::KeyCode::LCtrl) && PhysicsEngine::getKeyDown(input, PhysicsEngine::KeyCode::Y)) {
		executeCommand();
	}
}



void CommandManager::addCommand(Command* command)
{
	if (counter < commandHistory.size()) {
		commandHistory.resize(counter);
	}

	commandQueue.push(command);
}

void CommandManager::executeCommand()
{
	if (counter < commandHistory.size()) {
		commandHistory[counter]->execute(); // redo is tthe same as execute
		counter++;
	}
}

void CommandManager::undoCommand()
{
	if (counter > 0) {
		counter--;
		commandHistory[counter]->undo();
	}
}

bool CommandManager::canUndo()
{
	return counter > 0;
}

bool CommandManager::canRedo()
{
	return counter < commandHistory.size();
}