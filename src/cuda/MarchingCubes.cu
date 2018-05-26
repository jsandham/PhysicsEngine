//#include <iostream>
//
//#include <thrust/device_vector.h>
//#include <thrust/scan.h>
//
//#include "cuda_util.cuh"
//
//#include "MarchingCubes.cuh"
//#include "kernels.cuh"
//
//
//
//MarchingCubes::MarchingCubes(int numParticles, int3 marchingCubeGridDim, float3 marchingCubeGridSize)
//{
//	this->numParticles = numParticles;
//	this->marchingCubeGridDim = marchingCubeGridDim;
//	this->marchingCubeGridSize = marchingCubeGridSize;
//	numVoxels = marchingCubeGridDim.x*marchingCubeGridDim.y*marchingCubeGridDim.x;
//	numVoxelVertices = (marchingCubeGridDim.x + 1)*(marchingCubeGridDim.y + 1)*(marchingCubeGridDim.z + 1);
//	maxTriangles = 1000000;
//	elapsedTime = 0;
//
//	// allocate memory 
//	allocateMemory();
//}
//
//
//MarchingCubes::MarchingCubes(const MarchingCubes &mc)
//{
//	numParticles = mc.numParticles;
//	marchingCubeGridDim = mc.marchingCubeGridDim;
//	marchingCubeGridSize = mc.marchingCubeGridSize;
//	numVoxels = mc.numVoxels;
//	numVoxelVertices = mc.numVoxelVertices;
//	maxTriangles = mc.maxTriangles;
//	elapsedTime = mc.elapsedTime;
//
//	// allocate memory 
//	allocateMemory();
//}
//
//
//MarchingCubes& MarchingCubes::operator=(const MarchingCubes &mc)
//{
//	if(this != &mc){
//		// free old memory 
//		deallocateMemory();
//
//		numParticles = mc.numParticles;
//		marchingCubeGridDim = mc.marchingCubeGridDim;
//		marchingCubeGridSize = mc.marchingCubeGridSize;
//		numVoxels = mc.numVoxels;
//		numVoxelVertices = mc.numVoxelVertices;
//		maxTriangles = mc.maxTriangles;
//		elapsedTime = mc.elapsedTime;
//
//		// reallocate memory
//		allocateMemory();
//	}
//
//	return *this;
//}
//
//
//MarchingCubes::~MarchingCubes()
//{
//	// free memory 
//	deallocateMemory();
//}
//
//void MarchingCubes::allocateMemory()
//{
//	h_voxelOccupied = new int[numVoxels];
//	h_triangleVertices = new float[9*maxTriangles];
//	h_normals = new float[9*maxTriangles];
//
//	gpuErrchk(cudaMalloc((void**)&d_voxelVertexValues, numVoxelVertices*sizeof(float)));
//	gpuErrchk(cudaMalloc((void**)&d_triPerVoxel, numVoxels*sizeof(int)));
//	gpuErrchk(cudaMalloc((void**)&d_triPerVoxelScan, numVoxels*sizeof(int)));
//	gpuErrchk(cudaMalloc((void**)&d_voxelOccupied, numVoxels*sizeof(int)));
//	gpuErrchk(cudaMalloc((void**)&d_voxelOccupiedScan, numVoxels*sizeof(int)));
//}
//
//
//void MarchingCubes::deallocateMemory()
//{
//	delete [] h_voxelOccupied;
//	delete [] h_triangleVertices;
//	delete [] h_normals;
//
//	gpuErrchk(cudaFree(d_voxelVertexValues));
//	gpuErrchk(cudaFree(d_triPerVoxel));
//	gpuErrchk(cudaFree(d_triPerVoxelScan));
//	gpuErrchk(cudaFree(d_voxelOccupied));
//	gpuErrchk(cudaFree(d_voxelOccupiedScan));
//}
//
//
//void MarchingCubes::runMarchingCubesAlgorithm(float *voxelVertexValues, int numVoxelVertices, float isoValue)
//{
//	this->isoValue = isoValue;
//
//	elapsedTime = 0;
//	cudaEventCreate(&start);
//	cudaEventCreate(&stop);
//	cudaEventRecord(start,0);
//
//	gpuErrchk(cudaMemcpy(d_voxelVertexValues, voxelVertexValues, numVoxelVertices*sizeof(float), cudaMemcpyHostToDevice));
//
//	classifyVoxels();
//
//	thrust::exclusive_scan(thrust::device_ptr<int>(d_voxelOccupied),
//                           thrust::device_ptr<int>(d_voxelOccupied + numVoxels),
//                           thrust::device_ptr<int>(d_voxelOccupiedScan));
//
//	int lastElement, lastScanElement;
//    gpuErrchk(cudaMemcpy((void *) &lastElement,(void *)(d_voxelOccupied + numVoxels-1),sizeof(int), cudaMemcpyDeviceToHost));
//    gpuErrchk(cudaMemcpy((void *) &lastScanElement,(void *)(d_voxelOccupiedScan + numVoxels-1),sizeof(int), cudaMemcpyDeviceToHost));
//    numActiveVoxels = lastElement + lastScanElement;
//
//    gpuErrchk(cudaMalloc((void**)&d_compactVoxels, numActiveVoxels*sizeof(int)));
//
//    compactVoxels();
//
//    thrust::exclusive_scan(thrust::device_ptr<int>(d_triPerVoxel),
//                           thrust::device_ptr<int>(d_triPerVoxel + numVoxels),
//                           thrust::device_ptr<int>(d_triPerVoxelScan));
//
//    gpuErrchk(cudaMemcpy((void *) &lastElement,(void *)(d_triPerVoxel + numVoxels-1),sizeof(int), cudaMemcpyDeviceToHost));
//    gpuErrchk(cudaMemcpy((void *) &lastScanElement,(void *)(d_triPerVoxelScan + numVoxels-1),sizeof(int), cudaMemcpyDeviceToHost));
//    numTriangles = lastElement + lastScanElement;
//
//    gpuErrchk(cudaMalloc((void**)&d_triangleVertices, 9*numTriangles*sizeof(float)));
//    gpuErrchk(cudaMalloc((void**)&d_normals, 9*numTriangles*sizeof(float)));
//    //gpuErrchk(cudaMemset(d_triangleVertices, 9*numTriangles*sizeof(float), 0.0f));
//
//    generateTriangles();
//
//    gpuErrchk(cudaMemcpy(h_triangleVertices, d_triangleVertices, 9*numTriangles*sizeof(float), cudaMemcpyDeviceToHost));
//    gpuErrchk(cudaMemcpy(h_normals, d_normals, 9*numTriangles*sizeof(float), cudaMemcpyDeviceToHost));
//    // for(int i=0;i<9*numTriangles;i++){
//    // 	if(h_triangleVertices[i] > 1.0 || h_triangleVertices[i] < 0.0){
//    // 		std::cout << "error " << h_triangleVertices[i] << std::endl;
//    // 	}
//    // }
//
//	gpuErrchk(cudaMemcpy(h_voxelOccupied, d_voxelOccupied, numVoxels*sizeof(int), cudaMemcpyDeviceToHost));
//	std::cout << "number of active occupied voxels: " << numActiveVoxels << std::endl;
//	std::cout << "number of triangles: " << numTriangles << std::endl;
//
//	gpuErrchk(cudaFree(d_compactVoxels));
//	gpuErrchk(cudaFree(d_triangleVertices));
//	gpuErrchk(cudaFree(d_normals));
//
//	cudaEventRecord(stop,0);
//	cudaEventSynchronize(stop);
//	cudaEventElapsedTime(&elapsedTime, start, stop);
//}
//
//
//float* MarchingCubes::GetTriangleVertices()
//{
//	return h_triangleVertices;
//}
//
//float* MarchingCubes::GetTriangleNormals()
//{
//	return h_normals;
//}
//
//
//int MarchingCubes::GetNumberOfVertices()
//{
//	return 3*numTriangles;
//}
//
//
//void MarchingCubes::classifyVoxels()
//{
//	dim3 gridSize(256,1,1);
//	dim3 blockSize(256,1,1);
//	classify_voxels_kernel<<< gridSize, blockSize >>>
//		(
//			d_voxelVertexValues,
//			d_triPerVoxel, 
//			d_voxelOccupied,
//			numVoxels,
//			marchingCubeGridDim,
//			isoValue
//		);
//}
//
//
//void MarchingCubes::compactVoxels()
//{
//	dim3 gridSize(256,1,1);
//	dim3 blockSize(256,1,1);
//	compact_voxels_kernel<<< gridSize, blockSize >>>
//	(
//		d_compactVoxels,
//		d_voxelOccupied,
//		d_voxelOccupiedScan,
//		numVoxels
//	);
//}
//
//
//void MarchingCubes::generateTriangles()
//{
//	dim3 gridSize(256,1,1);
//	dim3 blockSize(256,1,1);
//	generate_triangles_kernel<<< gridSize, blockSize >>>
//	(
//		d_triangleVertices,
//		d_normals,
//		d_voxelVertexValues, 
//		d_compactVoxels,
//		d_triPerVoxelScan,
//		numVoxels,
//		numActiveVoxels,
//		marchingCubeGridDim,
//		marchingCubeGridSize,
//		isoValue
//	);
//}
//
//
//float MarchingCubes::GetIsovalue()
//{
//	return isoValue;
//}
//
//
//float MarchingCubes::GetElapsedTime()
//{
//	return elapsedTime;
//}
//
//
//size_t MarchingCubes::GetDeviceMemoryUsed()
//{
//	// we assume 4 byte int/float's
//	size_t numVoxelVerticesLengthArrays = 1 * 4 * numVoxelVertices;
//	size_t numVoxelsLengthArrays = 4 * 4 * numVoxels;
//
//	return numVoxelVerticesLengthArrays + numVoxelsLengthArrays;
//}
