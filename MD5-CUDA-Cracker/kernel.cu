#include "kernel.cuh"

__device__ bool found = false;

__global__ void crack_kernel(const uint hash[4], const char* alphabet, const uint alphabet_size, const uint min_size, const uint max_size, char* recovered)
{
	uint tid = threadIdx.x + blockIdx.x * blockDim.x, jump = blockDim.x * gridDim.x, current_size = min_size, result[4];

	uint* data = (uint*)malloc((max_size + 1) * sizeof(uint));
	memset(data, 0, (max_size + 1) * sizeof(uint));

	uchar* guess = (uchar*)malloc((max_size + 1) * sizeof(uchar));
	memset(guess, 0, (max_size + 1) * sizeof(uchar));

	if (!generate_next(data, current_size, tid, max_size, alphabet_size))
	{
		free(data);
		free(guess);

		return;
	}

	while (true)
	{
		for (int i = 0; i < current_size; i++)
		{
			guess[i] = alphabet[data[i]];
		}

		md5(guess, current_size, result);
		if (result[0] == hash[0] && result[1] == hash[1] && result[2] == hash[2] && result[3] == hash[3])
		{
			found = true;
			memcpy(recovered, guess, current_size);
			goto FREE;
		}

		for (int i = current_size; i < max_size; i++)
		{
			guess[i] = alphabet[0];

			md5(guess, i + 1, result);
			if (result[0] == hash[0] && result[1] == hash[1] && result[2] == hash[2] && result[3] == hash[3])
			{
				found = true;
				memcpy(result, guess, i);
				goto FREE;
			}
		}

		if (!generate_next(data, current_size, jump, max_size, alphabet_size) || found)
		{
			goto FREE;
		}
	}

FREE:
	free(data);
	free(guess);
}

__device__ bool generate_next(uint* data, uint& current_size, uint jump, const uint max_size, const uint alphabet_size)
{
	uint i = 0;

	while (jump != 0)
	{
		uint add = jump + data[i];
		data[i] = add % alphabet_size;
		jump = add / alphabet_size;

		if (++i > max_size)
		{
			return false;
		}
	}

	current_size = max(current_size, i);

	return true;
}

float run_kernel(int blocks, int threads, const uint hash[4], const char* alphabet, const size_t alphabet_size, uint min_size, uint max_size, char *recovered)
{
	cudaError_t cudaStatus;
	uint* dev_hash;

	if ((cudaStatus = cudaMalloc((void**)&dev_hash, 4 * sizeof(uint))) != cudaSuccess)
	{
		std::cerr << "cudaMalloc dev_hash failed!" << std::endl;
		error({ dev_hash });
	}

	if ((cudaStatus = cudaMemcpy(dev_hash, hash, 4 * sizeof(uint), cudaMemcpyHostToDevice)) != cudaSuccess)
	{
		std::cerr << "cudaMemcpy dev_hash failed!" << std::endl;
		error({ dev_hash });
	}

	char* dev_alphabet;

	if ((cudaStatus = cudaMalloc((void**)&dev_alphabet, alphabet_size * sizeof(char))) != cudaSuccess)
	{
		std::cerr << "cudaMalloc dev_alphabet failed!" << std::endl;
		error({ dev_hash, dev_alphabet });
	}

	if ((cudaStatus = cudaMemcpy(dev_alphabet, alphabet, alphabet_size * sizeof(char), cudaMemcpyHostToDevice)) != cudaSuccess)
	{
		std::cerr << "cudaMemcpy dev_alphabet failed!" << std::endl;
		error({ dev_hash, dev_alphabet });
	}

	char* dev_recovered;

	if ((cudaStatus = cudaMalloc((void**)&dev_recovered, (max_size + 1) * sizeof(char))) != cudaSuccess)
	{
		std::cerr << "cudaMalloc dev_recovered failed!" << std::endl;
		error({ dev_hash, dev_alphabet, dev_recovered });
	}

	if ((cudaStatus = cudaMemset(dev_recovered, 0, (max_size + 1) * sizeof(char))) != cudaSuccess)
	{
		std::cerr << "cudaMemset dev_recovered failed!" << std::endl;
		error({ dev_hash, dev_alphabet, dev_recovered });
	}

	cudaEvent_t start, stop;

	if ((cudaStatus = cudaEventCreate(&start)) != cudaSuccess)
	{
		std::cerr << "cudaEventCreate start failed!" << std::endl;
		error({ dev_hash, dev_alphabet, dev_recovered });
	}

	if ((cudaStatus = cudaEventCreate(&stop)) != cudaSuccess)
	{
		std::cerr << "cudaEventCreate stop failed!" << std::endl;
		error({ dev_hash, dev_alphabet, dev_recovered });
	}

	if ((cudaStatus = cudaEventRecord(start, 0)) != cudaSuccess)
	{
		std::cerr << "cudaEventRecord start failed!" << std::endl;
		error({ dev_hash, dev_alphabet, dev_recovered });
	}

	crack_kernel << <blocks, threads >> > (dev_hash, dev_alphabet, alphabet_size, min_size, max_size, dev_recovered);

	if ((cudaStatus = cudaGetLastError()) != cudaSuccess)
	{
		std::cerr << "crack_kernel launch failed: " << cudaGetErrorString(cudaStatus) << std::endl;
		error({ dev_hash, dev_alphabet, dev_recovered });
	}

	if ((cudaStatus = cudaDeviceSynchronize()) != cudaSuccess)
	{
		std::cout << "cudaDeviceSynchronize returned error code " << cudaStatus << " after launching crack_kernel!" << std::endl;
		error({ dev_hash, dev_alphabet, dev_recovered });
	}

	if ((cudaStatus = cudaEventRecord(stop, 0)) != cudaSuccess)
	{
		std::cerr << "cudaEventRecord stop failed!" << std::endl;
		error({ dev_hash, dev_alphabet, dev_recovered });
	}

	if ((cudaStatus = cudaMemcpy(recovered, dev_recovered, max_size + 1 * sizeof(char), cudaMemcpyDeviceToHost)) != cudaSuccess)
	{
		std::cerr << "cudaMemcpy dev_recovered failed!" << std::endl;
		error({ dev_hash, dev_alphabet });
	}

	float elapsed;

	if ((cudaStatus = cudaEventElapsedTime(&elapsed, start, stop)) != cudaSuccess)
	{
		std::cerr << "cudaEventElapsedTime failed!" << std::endl;
		error({ dev_hash, dev_alphabet, dev_recovered });
	}

	free_dev({ dev_hash, dev_alphabet, dev_recovered });

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	return elapsed;
}

void error(std::initializer_list<void*> devs)
{
	free_dev(devs);

	std::exit(1);
}

void free_dev(std::initializer_list<void*> devs)
{
	for (auto& dev : devs)
	{
		cudaFree(dev);
	}
}
