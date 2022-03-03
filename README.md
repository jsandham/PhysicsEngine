# PhysicsEngine 

## Getting Started
I use Visual Studio 2022 but use whatever editor works best for you. I use the 'cl' C++ compiler that comes with Visual Studio. You can use a different c++ compiler but this will require you to modify the bat scripts.

<ins>**1. Downloading the repository:**</ins>

Start by cloning the repository with `https://github.com/jsandham/PhysicsEngine.git`.

<ins>**2. Configuring and building:**</ins>

1. Edit the [shell.bat](https://github.com/jsandham/PhysicsEngine/blob/master/shell.bat) file found at the root of the repository to point to whereever your vcvarsall.bat script is located. Typically this is located in something like C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\ but this will vary on your visual studio installation.
2. Run the [build_all_debug.bat](https://github.com/jsandham/PhysicsEngine/blob/master/build_all_debug.bat) for a debug build or [build_all_release.bat](https://github.com/jsandham/PhysicsEngine/blob/master/build_all_release.bat) for a release build. These bat scipts will build all dependencies, the engine and the editor.
3. You can now run the [editor](https://github.com/jsandham/PhysicsEngine/tree/master/editor/bin/debug) executable.
4. Thats it.

<ins>**3. Dependencies:**</ins>

All of the dependencies are already included as part of the repository and are located in the [external](https://github.com/jsandham/PhysicsEngine/tree/master/external) folder and are rebuilt when the engine is built. One of my philosophies with this engine is to 1) limit the number of dependencies, and 2) dont force users (or myself) to have to install spurious extra software in order to use my engine. What this means is that all you need is a C++ compiler. Any dependencies should be included as part of the repository and are built from source. No installing cmake. No installing python. No installing chocolatey. No powershell. It also means that all of the files found in the repository should be files that the average C++ programmer can understand. Because I don't use a build system (only simple bat files), there are no solution files or other extraneous Visual Studio cruft. There are no CmakeList files sprinkled throughout the project. The ultimate goal here is to have the user be able to 1) clone the repo and 2) run a simple bat file. I will continue to test on various Windows machines to fix problems encountered when trying to build this engine.

***

## What is this?
This repo contains my attempt at making a game engine written in C++ and opengl. This includes an [engine](https://github.com/jsandham/PhysicsEngine/tree/master/engine) library and an gui [editor](https://github.com/jsandham/PhysicsEngine/tree/master/editor). The editor uses the engine library and imgui (see [external](https://github.com/jsandham/PhysicsEngine/tree/master/engine) folder for all dependencies). Currently my engine is just called 'PhysicsEngine' because originally I was looking to just make a simple physics engine but once I come up with a good name ill change this. 

## What is the plan?
I am developing this as a hobby as a way to learn about rendering, physics, I/O, serialization, GUI programming, C++, and anything else I run into while developing it. Right now I am working on basic features, so the engine is not something you would use to actually make a game...yet =).

![PhysicsEngine](/resources/snapshots/editor_snapshot.PNG?raw=true "PhysicsEngine") 

## What can you do with the engine?
Not much right now. Currently I am still implementing core features like loading assets, creating materials, and testing rendering. As I make progress ill keep updating this readme. Right now I am working on a terrain generation system but I also need to implement physics still as well as fine tune my material system to increase its flexibility.

![PhysicsEngine](/resources/snapshots/editor_snapshot2.PNG?raw=true "PhysicsEngine") 