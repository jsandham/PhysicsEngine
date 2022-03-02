# PhysicsEngine 

## Getting Started
I use visual studio 2022 but use whatever editor works best for you. I use the 'cl' c++ compiler that comes with visual studio. You can use a different c++ compiler but this will require you to modify the bat scripts.

<ins>**1. Downloading the repository:**</ins>

Start by cloning the repository with `https://github.com/jsandham/PhysicsEngine.git`.

<ins>**2. Configuring the dependencies:**</ins>

1. Edit the [shell.bat](https://github.com/jsandham/PhysicsEngine/blob/master/shell.bat) file found root folder to point to whereever your vcvarsall.bat script is located. Typically this is located in something like C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\ but this will vary on your visual studio installation.
2. Run the [build_all_debug.bat](https://github.com/jsandham/PhysicsEngine/blob/master/build_all_debug.bat) for a debug build or [build_all_release.bat](https://github.com/jsandham/PhysicsEngine/blob/master/build_all_release.bat) for a release build. These bat scipts will build all dependencies, the engine and the editor.
3. Thats it. 

All of the dependencies are already included as part of the repo and are located in the [external](https://github.com/jsandham/PhysicsEngine/tree/master/external) folder and are rebuilt when the engine is built. 

***

## What is this?
This repo contains my attempt at making a game engine written in C++ and opengl. This includes a engine library [engine](https://github.com/jsandham/PhysicsEngine/tree/master/engine) and an gui editor [editor](https://github.com/jsandham/PhysicsEngine/tree/master/editor). The editor uses the engine library and imgui (see [external](https://github.com/jsandham/PhysicsEngine/tree/master/engine) folder for all dependencies). Currently my engine is just called 'PhysicsEngine' because originally I was looking to just make a simple physics engine but once I come up with a good name ill change this. 

## What is the plan?
I am developing this as a hobby as a way to learn about rendering, physics, I/O, serialization, GUI programming, C++, and anything else I run into while eveloping it. Right now I am working on basic features, so the engine is not something you would use to actually make a game...yet =).

![PhysicsEngine](/resources/snapshots/editor_snapshot.PNG?raw=true "PhysicsEngine") 
