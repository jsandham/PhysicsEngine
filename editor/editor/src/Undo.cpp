#include "../include/Undo.h"

#include "imgui.h"

#include "core/Log.h"

using namespace PhysicsEditor;

std::vector< std::pair<PhysicsEngine::Entity*, std::vector<char>>> Undo::mCreatedEntityRecords;
std::vector< std::pair<PhysicsEngine::Entity*, std::vector<char>>> Undo::mEntityRecords;
std::vector< std::pair<PhysicsEngine::Component*, std::vector<char>>> Undo::mAddComponentRecords;
std::vector< std::pair<PhysicsEngine::Component*, std::vector<char>>> Undo::mComponentRecords;

int Undo::counter = 0;
std::vector<Command*> Undo::commandHistory;
std::queue<Command*> Undo::commandQueue;

void Undo::updateUndoStack(EditorClipboard& clipboard)
{
	for (size_t i = 0; i < mCreatedEntityRecords.size(); i++)
	{
		Undo::addCommand(new RecordEntityCreationCommand(clipboard.getWorld(), mCreatedEntityRecords[i].first->getId(), mCreatedEntityRecords[i].second));
	}

	for (size_t i = 0; i < mAddComponentRecords.size(); i++)
	{
		//Undo::addCommand(new RecordComponentAdditionCommand(clipboard.getWorld(), mAddComponentRecords[i].first->getId(), mAddComponentRecords[i].second));
	}

	for (size_t i = 0; i < mEntityRecords.size(); i++)
	{
		std::vector<char> temp = mEntityRecords[i].first->serialize();

		if (mEntityRecords[i].second != temp)
		{
			Undo::addCommand(new RecordObjectCommand(mEntityRecords[i].first, mEntityRecords[i].second, temp));
		}
	}

	for (size_t i = 0; i < mComponentRecords.size(); i++)
	{
		std::vector<char> temp = mComponentRecords[i].first->serialize();

		if (mComponentRecords[i].second != temp)
		{
			Undo::addCommand(new RecordObjectCommand(mComponentRecords[i].first, mComponentRecords[i].second, temp));
		}
	}

	mCreatedEntityRecords.clear();
	mEntityRecords.clear();
	mAddComponentRecords.clear();
	mComponentRecords.clear();

	//PhysicsEngine::Log::info(("command history: " + std::to_string(commandHistory.size()) + "counter: " + std::to_string(counter) + "\n").c_str());

	while (!commandQueue.empty())
	{
		Command* command = commandQueue.front();
		commandQueue.pop();

		command->execute();

		commandHistory.push_back(command);
		counter++;
	}

	ImGuiIO& io = ImGui::GetIO();

	if (io.KeysDown[17] && ImGui::IsKeyPressed(90) /*LCtrl + Z*/)
	{
		undoCommand();
	}

	if (io.KeysDown[17] && ImGui::IsKeyPressed(89) /*LCtrl + Y*/)
	{
		executeCommand();
	}
}

void Undo::clearUndoStack()
{
	// delete any remaining commands in queue and history
    for (size_t i = 0; i < commandHistory.size(); i++)
    {
        delete commandHistory[i];
    }

    while (!commandQueue.empty())
    {
        Command *command = commandQueue.front();
        commandQueue.pop();

        delete command;
    }
}

void Undo::recordEntityCreation(PhysicsEngine::Entity* entity)
{
	Undo::mCreatedEntityRecords.push_back(std::make_pair(entity, entity->serialize()));
}

void Undo::recordEntity(PhysicsEngine::Entity* entity)
{
	Undo::mEntityRecords.push_back(std::make_pair(entity, entity->serialize()));
}

void Undo::recordComponent(PhysicsEngine::Component* component)
{
	Undo::mComponentRecords.push_back(std::make_pair(component, component->serialize()));
}

void Undo::addCommand(Command* command)
{
	for (size_t i = counter; i < commandHistory.size(); i++)
	{
		delete commandHistory[i];
	}

	commandHistory.resize(counter);

	commandQueue.push(command);
}

void Undo::executeCommand()
{
	if (counter < commandHistory.size())
	{
		commandHistory[counter]->execute(); // redo is the same as execute
		counter++;
	}
}

void Undo::undoCommand()
{
	if (counter > 0)
	{
		counter--;
		commandHistory[counter]->undo();
	}
}

bool Undo::canUndo()
{
	return counter > 0;
}

bool Undo::canRedo()
{
	return counter < commandHistory.size();
}