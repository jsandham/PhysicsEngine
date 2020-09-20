PhysicsEngine
=============
An engine for games and physics simulations on windows using OpenGL. 

Engine
======

Overview
--------

The engine source code is located in the /src and /include directories

Compilation
-----------

To compile the engine static library:
* navigate to engine/lib build folder
* run build_engine.bat

By default this builds the engine static library in debug mode. You can specify 
the build mode as /debug or /release:
* build_engine.bat <build mode>

The static library (engine.lib) should then be located in build/Debug or build/Release

Editor
======

Overview
--------

The editor is built as a visual studio project which links with the engine static library. 
To use the editor simply navigate to editor/x64/debug or editor/x64/release folder and run 
executable. You can also rebuild the editor by opening the visual studio project and rebuilding from there. 