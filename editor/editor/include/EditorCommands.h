#ifndef __EDITOR_COMMANDS_H__
#define __EDITOR_COMMANDS_H__

#include "Command.h"

#include "core/Guid.h"
#include "core/World.h"
#include "components/Transform.h"
#include "components/Rigidbody.h"

namespace PhysicsEditor
{
	template<class T>
	class ChangePropertyCommand : public Command
	{
	private:
		T* valuePtr;
		T oldValue;
		T newValue;

	public:
		ChangePropertyCommand(T* valuePtr, T newValue)
		{
			this->valuePtr = valuePtr;
			this->oldValue = *valuePtr;
			this->newValue = newValue;
		}

		void execute()
		{
			*valuePtr = newValue;
		}

		void undo()
		{
			*valuePtr = oldValue;
		}
	};

	// Hierarchy commands

	class CreateEntityCommand : public Command
	{
		private:
			PhysicsEngine::World* world;
			PhysicsEngine::Guid entityId;

		public:
			CreateEntityCommand(PhysicsEngine::World* world);

			void execute() override;
			void undo() override;
	};

	class CreateCameraCommand : public Command
	{
		private:
			PhysicsEngine::World* world;
			PhysicsEngine::Guid entityId;

		public:
			CreateCameraCommand(PhysicsEngine::World* world);

			void execute() override;
			void undo() override;
	};

	class CreateLightCommand : public Command
	{
		private:
			PhysicsEngine::World* world;
			PhysicsEngine::Guid entityId;

		public:
			CreateLightCommand(PhysicsEngine::World* world);

			void execute() override;
			void undo() override;
	};

	class CreateCubeCommand : public Command
	{
		private:
			PhysicsEngine::World* world;
			PhysicsEngine::Guid entityId;

		public:
			CreateCubeCommand(PhysicsEngine::World* world);

			void execute() override;
			void undo() override;
	};

	class CreateSphereCommand : public Command
	{
		private:
			PhysicsEngine::World* world;
			PhysicsEngine::Guid entityId;

		public:
			CreateSphereCommand(PhysicsEngine::World* world);

			void execute() override;
			void undo() override;
	};
}

#endif
