//#ifndef __MARCHINGCUBES_H__
//#define __MARCHINGCUBES_H__
//
//class MarchingCubes
//{
//	private:
//		int numParticles;
//		int numVoxels;
//		int numActiveVoxels;
//		int numVoxelVertices;
//		int numTriangles;
//		int maxTriangles;
//		int3 marchingCubeGridDim;
//		float3 marchingCubeGridSize;
//
//		float isoValue;
//
//		// host arrays
//		int *h_voxelOccupied;
//		int *h_voxelOccupiedScan;
//		float *h_triangleVertices;
//		float *h_normals;
//
//		// device arrays
//		float *d_voxelVertexValues;
//		int *d_triPerVoxel;
//		int *d_triPerVoxelScan;
//		int *d_voxelOccupied;
//		int *d_voxelOccupiedScan;
//		int *d_compactVoxels;
//		float *d_triangleVertices;
//		float *d_normals;
//
//		// used for timing
//		float elapsedTime;
//		cudaEvent_t start, stop;
//		
//
//	public:
//		MarchingCubes(int numParticles, int3 marchingCubeGridDim, float3 marchingCubeGridSize);
//		MarchingCubes(const MarchingCubes &mc);
//		MarchingCubes& operator=(const MarchingCubes &mc);
//		~MarchingCubes();
//
//		void runMarchingCubesAlgorithm(float *voxelVertexValues, int numVoxelVertices, float isoValue);
//
//		float* GetTriangleVertices();
//		float* GetTriangleNormals();
//		int GetNumberOfVertices();
//		float GetIsovalue();
//		float GetElapsedTime();
//		size_t GetDeviceMemoryUsed();
//
//	private:
//		void allocateMemory();
//		void deallocateMemory();
//		void classifyVoxels();
//		void compactVoxels();
//		void generateTriangles();
//};
//
//#endif