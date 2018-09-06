// #ifndef __SCENE_H__
// #define __SCENE_H__

// #include <vector>
// #include <string>
// #include <map>

// #include "Manager.h"

// #include "../systems/PlayerSystem.h"
// #include "../systems/PhysicsSystem.h"
// #include "../systems/RenderSystem.h"

// namespace PhysicsEngine
// {
// 	class Scene
// 	{
// 		public:
// 			std::string name;
// 			std::string filepath;
// 			bool isLoaded;

// 		public:
// 			Scene();
// 			~Scene();
// 	};





	// class Scene
	// {
	// 	private:
	// 		std::string sceneName;
	// 		Manager manager;
	// 		PlayerSystem *playerSystem;
	// 		PhysicsSystem *physicsSystem;
	// 		RenderSystem *renderSystem;

	// 	public:
	// 		Scene();
	// 		~Scene();

	// 		bool validate(std::string sceneFilePath, std::vector<std::string> assetFilePaths);
	// 		void load(std::string sceneFilePath, std::vector<std::string> assetFilePaths);
	// 		void init();
	// 		void update();
	// };
}











// #include <map>
// #include <string>
// #include <ctime>
// #include <vector>

// #include <SFML/Window.hpp>
// #include <SFML/Graphics.hpp>

// #define GLM_FORCE_RADIANS

// #include "glm/glm.hpp"

// #include "systems/System.h"
// #include "entities/Entity.h"
// #include "components/Camera.h"


// namespace PhysicsEngine
// {
// 	class Scene
// 	{
// 		private:
// 			sf::RenderWindow *window;
		
// 			std::vector<std::string> systemKeys;
// 			std::vector<System*> systems;

// 			sf::Font font;
// 			sf::Text timeText;
// 			sf::Text versionText;

// 			std::string sceneName;

// 			glm::vec3 cameraPos;
// 			glm::vec3 cameraFront;
// 			glm::vec3 cameraUp;

// 			clock_t startTime;
// 			clock_t PhysicsEngineEndTime;
// 			clock_t renderStartTime;
// 			clock_t endTime;

// 			bool showLights;

// 			int openglFrame;
// 			int totalElapsedFrames;
// 			double totalElapsedTime;
// 			double timePerFrame;
// 			double PhysicsEngineTimePerFrame;
// 			double renderTimePerFrame;
// 			double fps;

// 		public:
// 			Scene(sf::RenderWindow *window);
// 			~Scene();

// 			void run();
// 			void addSystem(std::string key, System *system);
// 			void removeSystem(std::string key);

// 			void setTimeText(std::string text);
// 			void setVersionText(std::string text);
// 			void setSceneName(std::string name);

// 		private:
// 			void setDefaultConfig();

// 			void init();
// 			void update();
// 			void updateTimer();
// 			void processEvents();

// 			bool isWindowOpen();
// 	};
// }

// #endif