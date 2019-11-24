#ifndef __EDITOR_COMMANDS_H__
#define __EDITOR_COMMANDS_H__

#include "Command.h"

#include "core/Guid.h"
#include "core/World.h"
#include "components/Transform.h"
#include "components/Rigidbody.h"

namespace PhysicsEditor
{
	// Hierarchy commands

	class CreateEntityCommand : public Command
	{
		private:
			PhysicsEngine::World* world;
			std::vector<char> entityData;

		public:
			CreateEntityCommand(PhysicsEngine::World* world);

			void execute() override;
			void undo() override;
	};

	class CreateCameraCommand : public Command
	{
		private:
			PhysicsEngine::World* world;
			std::vector<char> entityData;
			std::vector<char> cameraData;

		public:
			CreateCameraCommand(PhysicsEngine::World* world);

			void execute() override;
			void undo() override;
	};

	class CreateLightCommand : public Command
	{
		private:
			PhysicsEngine::World* world;
			std::vector<char> entityData;
			std::vector<char> lightData;

		public:
			CreateLightCommand(PhysicsEngine::World* world);

			void execute() override;
			void undo() override;
	};

	class CreateCubeCommand : public Command
	{
		private:
			PhysicsEngine::World* world;
			std::vector<char> entityData;
			std::vector<char> boxColliderData;
			std::vector<char> meshRendererData;

		public:
			CreateCubeCommand(PhysicsEngine::World* world);

			void execute() override;
			void undo() override;
	};

	class CreateSphereCommand : public Command
	{
		private:
			PhysicsEngine::World* world;
			std::vector<char> entityData;
			std::vector<char> sphereColliderData;
			std::vector<char> meshRendererData;

		public:
			CreateSphereCommand(PhysicsEngine::World* world);

			void execute() override;
			void undo() override;
	};

	class DestroyEntityCommand : public Command
	{
		private:
			PhysicsEngine::World* world;
			std::vector<char> entityData;
			std::vector<std::pair<int, std::vector<char>>> components;

		public:
			DestroyEntityCommand(PhysicsEngine::World* world, PhysicsEngine::Guid entityId);

			void execute() override;
			void undo() override;
	};

	// inspector commands

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

	template<class T>
	class AddComponentCommand : public Command
	{
		private:
			PhysicsEngine::World* world;
			PhysicsEngine::Guid entityId;
			PhysicsEngine::Guid componentId;

		public:
			AddComponentCommand(PhysicsEngine::World* world, PhysicsEngine::Guid entityId)
			{
				this->world = world;
				this->entityId = entityId;
			}

			void execute()
			{
				Entity* entity = world->getEntity(entityId);
				T* component = entity->addComponent<T>(world);
				componentId = component->componentId;
			}

			void undo()
			{
				world->latentDestroyComponent(entityId, componentId, ComponentType<T>::type);
			}
	};

	//template<class T>
	//class ChangeComponentValue
}

#endif
