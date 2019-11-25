@echo off

if not defined DevEnvDir (
	call "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvarsall.bat" x64
	rem call "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars32.bat" x64
)

rem dumpbin /SYMBOLS "C:\Users\jsand\Documents\PhysicsEngine\engine\lib\obj\Color.obj" > "dumpbin.txt"
dumpbin /DISASM "C:\Users\jsand\Documents\PhysicsEngine\engine\lib\obj\Color.obj" > "dumpbin.txt"