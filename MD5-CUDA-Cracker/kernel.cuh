#include <iostream>
#include "md5.cuh"

__global__ void crack_kernel(const uint hash[4], const char* alphabet, const uint alphabet_size, const uint min_size, const uint max_size);

__device__ bool generate_next(uint* data, uint& current_size, uint jump, const uint max_size, const uint alphabet_size);

float run_kernel(int blocks, int threads, const uint hash[4], const char* alphabet, const size_t alphabet_size, uint min_size, uint max_size, char* recovered);

void error(std::initializer_list<void*> devs);

void free_dev(std::initializer_list<void*> devs);
