#ifndef __DEBUG_H__
#define __DEBUG_H__

#include <string>


#define DEBUG 1
#define TIMING 0

#define DEBUG_PRINT(stream, statement) \
	do { if(DEBUG) (stream) << "DEBUG: "<< __FILE__<<"("<<__LINE__<<") " << (statement) << std::endl;} while(0)

#define TIMING_PRINT(stream, statement, time) \
	do { if(TIMING) (stream) << "TIMING: "<<__FILE__<<"("<<__LINE__<<") " << (statement) << (time) << std::endl;} while(0)

int DEBUG_TEST_MATRIX_AMG(std::string filename, unsigned int nnz, unsigned int nr, unsigned int solver);


#endif