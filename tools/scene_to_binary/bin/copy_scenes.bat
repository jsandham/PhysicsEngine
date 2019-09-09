@echo off

for /R C:\Users\jsand\Documents\PhysicsEngine\tools\scene_to_binary\bin\scenes %%f in (*.data) do copy %%f C:\Users\jsand\Documents\PhysicsEngine\sample_project\Demo\x64\Debug\scenes
for /R C:\Users\jsand\Documents\PhysicsEngine\tools\scene_to_binary\bin\assets %%f in (*.data) do copy %%f C:\Users\jsand\Documents\PhysicsEngine\sample_project\Demo\x64\Debug\assets

for /R C:\Users\jsand\Documents\PhysicsEngine\tools\scene_to_binary\bin\scenes %%f in (*.data) do copy %%f C:\Users\jsand\Documents\PhysicsEngine\sample_project\Demo\x64\Release\scenes
for /R C:\Users\jsand\Documents\PhysicsEngine\tools\scene_to_binary\bin\assets %%f in (*.data) do copy %%f C:\Users\jsand\Documents\PhysicsEngine\sample_project\Demo\x64\Release\assets