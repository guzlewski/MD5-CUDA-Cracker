﻿#include <cuda_runtime.h>
#include <cstdint>

#define uint uint32_t
#define uchar uint8_t

#define block_size 64

#define S11 7
#define S12 12
#define S13 17
#define S14 22
#define S21 5
#define S22 9
#define S23 14
#define S24 20
#define S31 4
#define S32 11
#define S33 16
#define S34 23
#define S41 6
#define S42 10
#define S43 15
#define S44 21

#define F(x, y, z) (((x) & (y)) | ((~x) & (z)))
#define G(x, y, z) (((x) & (z)) | ((y) & (~z)))
#define H(x, y, z) ((x) ^ (y) ^ (z))
#define I(x, y, z) ((y) ^ ((x) | (~z)))

#define ROTATE_LEFT(x, n) (((x) << (n)) | ((x) >> (32 - (n))))

#define FF(a, b, c, d, x, s, ac)                    \
	{                                               \
		(a) += F((b), (c), (d)) + (x) + (uint)(ac); \
		(a) = ROTATE_LEFT((a), (s));                \
		(a) += (b);                                 \
	}

#define GG(a, b, c, d, x, s, ac)                    \
	{                                               \
		(a) += G((b), (c), (d)) + (x) + (uint)(ac); \
		(a) = ROTATE_LEFT((a), (s));                \
		(a) += (b);                                 \
	}

#define HH(a, b, c, d, x, s, ac)                    \
	{                                               \
		(a) += H((b), (c), (d)) + (x) + (uint)(ac); \
		(a) = ROTATE_LEFT((a), (s));                \
		(a) += (b);                                 \
	}

#define II(a, b, c, d, x, s, ac)                    \
	{                                               \
		(a) += I((b), (c), (d)) + (x) + (uint)(ac); \
		(a) = ROTATE_LEFT((a), (s));                \
		(a) += (b);                                 \
	}

__device__ __host__ uint byteswap(uint word);

__device__ __host__ void transform(uint state[4], const uchar block[block_size]);

__device__ __host__ void md5(const uchar* data, const uint size, uint result[4]);
