#include <iostream>
#include <unordered_map>

#include "argparse.hpp"
#include "kernel.cuh"

std::string remove_duplicates(const std::string& str)
{
	std::string result = "";
	std::unordered_map<char, bool> exists;

	for (const auto& letter : str)
	{
		if (!exists[letter])
		{
			exists[letter] = true;
			result += letter;
		}
	}

	return result;
}

uint* parse_hash(const std::string& str)
{
	uint* hash = new uint[4];

	if (sscanf_s(str.c_str(), "%8x%8x%8x%8x", &hash[0], &hash[1], &hash[2], &hash[3]) != 4)
	{
		delete[] hash;
		throw std::runtime_error("Invalid --hash format.");
	}

	for (int i = 0; i < 4;i++)
	{
		hash[i] = byteswap(hash[i]);
	}

	return hash;
}

void validate_arguments(int min_size, int max_size)
{
	if (min_size < 1)
	{
		throw std::runtime_error("Invalid --min value.");
	}

	if (max_size < 1)
	{
		throw std::runtime_error("Invalid --max value.");
	}

	if (min_size > max_size)
	{
		throw std::runtime_error("--max value must be greater or equal --min");
	}
}

bool validate_result(uint hash[4], char* recovered)
{
	uint result[4] = { 0 };
	md5((uchar*)recovered, strlen(recovered), result);

	return hash[0] == result[0] && hash[1] == result[1] && hash[2] == result[2] && hash[3] == result[3];
}

int main(int argc, char* argv[])
{
	argparse::ArgumentParser program;

	program.add_argument("hash")
		.help("input md5 hash to crack ")
		.action([](const std::string& value) { return parse_hash(value); })
		.required();

	program.add_argument("alphabet")
		.help("input alphabet to generate combinations ")
		.action([](const std::string& value) { return remove_duplicates(value); })
		.required();

	program.add_argument("--min")
		.help("starting lenght of combinations, 1 if not supplied ")
		.action([](const std::string& value) { return std::stoi(value); })
		.default_value(1);

	program.add_argument("--max")
		.help("maximum lenght of combinations, 8 if not supplied ")
		.action([](const std::string& value) { return std::stoi(value); })
		.default_value(8);

	try
	{
		program.parse_args(argc, argv);
		validate_arguments(program.get<int>("--min"), program.get<int>("--max"));
	}
	catch (const std::runtime_error& err)
	{
		std::cerr << err.what() << std::endl;
		std::cout << program;
		return 1;
	}

	uint* hash = program.get<uint*>("hash");
	std::string alphabet = program.get<std::string>("alphabet");

	int min_size = program.get<int>("--min");
	int max_size = program.get<int>("--max");

	std::cout << "Cracking hash " << std::hex << byteswap(hash[0]) << byteswap(hash[1]) << byteswap(hash[2]) << byteswap(hash[3]) << std::dec << std::endl
		<< "with alphabet " << alphabet << std::endl
		<< "alphabet length " << alphabet.length() << std::endl
		<< "combination minimum size " << min_size << std::endl
		<< "combination maximum size " << max_size << std::endl;

	char* recovered = new char[max_size + 1];

	float kernel_time = run_kernel(2048, 512, hash, alphabet.c_str(), alphabet.length(), min_size, max_size, recovered);

	if (strlen(recovered) != 0)
	{
		std::cout << "Found matching combination in " << kernel_time / 1000.0 << " s." << std::endl
			<< "Recovered data: " << recovered << std::endl;

		if (validate_result(hash, recovered))
		{
			std::cout << "Result is valid." << std::endl;
		}
		else
		{
			std::cout << "Result is invalid." << std::endl;
		}
	}
	else
	{
		std::cout << "Checked all combinations in " << kernel_time / 1000.0 << " s." << std::endl
			<< "No matching combination was found." << std::endl;
	}

	delete[] hash;
	delete[] recovered;
	return 0;
}
