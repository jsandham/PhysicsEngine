#include "../include/CommandManager.h"

#include "../include/imgui/imgui.h"

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

void CommandManager::update()
{
	while (!commandQueue.empty()) {
		Command* command = commandQueue.front();
		commandQueue.pop();

		command->execute();

		commandHistory.push_back(command);
		counter++;
	}

	ImGuiIO& io = ImGui::GetIO();

	if (io.KeysDown[17] && ImGui::IsKeyPressed(90) /*LCtrl + Z*/) {
		undoCommand();
	}

	if (io.KeysDown[17] && ImGui::IsKeyPressed(89) /*LCtrl + Y*/) {
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