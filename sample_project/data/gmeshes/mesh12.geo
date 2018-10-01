Point(1) = {-1, 0.6, 0, 1.0};
Point(2) = {-1, -0.1, 0, 1.0};
Point(3) = {-0.6, -0.1, 0, 1.0};
Point(4) = {-0.6, 0.6, 0, 1.0};
Line(1) = {2, 3};
Line(2) = {3, 4};
Line(3) = {4, 1};
Line(4) = {1, 2};
Line Loop(5) = {2, 3, 4, 1};
Plane Surface(6) = {5};
Extrude {0, 0, 10} {
  Surface{6};
}
Physical Surface(29) = {6, 23, 15, 27, 19};
Physical Surface(30) = {28};
Physical Volume(31) = {1};
