#include "utility.h"
#include <Windows.h>
#include <iostream>
#include <cstdio>
#include <cstdlib>

EmbedingMatrix::EmbedingMatrix(int _word_table_size, int _embeding_size, bool _is_sparse)
{
	word_table_size = _word_table_size;
	embeding_size = _embeding_size;
	is_sparse = _is_sparse;
	sparse = NULL; 
	if (is_sparse)
	{
		feat_table.clear();
		for (int i = 0; i < _embeding_size; ++i)
			feat_table.push_back(std::vector<FeatPair>(0));
	} else 
	{
		dense.resize(embeding_size, word_table_size);
	}
}

void EmbedingMatrix::Destroy()
{
	if (is_sparse)
	{
		for (int i = 0; i < word_table_size; ++i)
			delete[] sparse[i];
	}
}

void EmbedingMatrix::GetSparseInnerProducts(int word_idx, int base, ele_type* innter_products)
{
	memset(innter_products + base, 0, word_table_size * sizeof(ele_type));
	int feat_idx, v_idx;
	ele_type feat_value;
	for (int i = 0; sparse[word_idx][i].idx != INT_MAX; ++i)
	{
		feat_idx = sparse[word_idx][i].idx;
		feat_value = sparse[word_idx][i].value;
		for (size_t j = 0; j < feat_table[feat_idx].size(); ++j)
		{
			v_idx = feat_table[feat_idx][j].idx;
			innter_products[base + v_idx] += feat_value * feat_table[feat_idx][j].value;
		}
	}
}

FileReader::FileReader()
{
	content = 0;
	file_size = 0;
	cur_pos = 0;
}

void FileReader::OpenFile(const char* filename)
{
	HANDLE hFile = CreateFileA(filename, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
	if (hFile == INVALID_HANDLE_VALUE)
		return;
	LARGE_INTEGER size;
	if (GetFileSizeEx(hFile, &size) == 0)
	{
		CloseHandle(hFile);
		return;
	}
	file_size = size.QuadPart;
	content = (char *) malloc(file_size);

	CloseHandle(hFile);


	FILE* fid = fopen(filename, "rb");
	fread(content, sizeof(char), file_size, fid);
	fclose(fid);
	cur_pos = 0;
}

void FileReader::SkipWhiteSpace()
{
	while (cur_pos < file_size && IsWhiteSpace(content[cur_pos]))
		cur_pos++;
}

int FileReader::ReadInt()
{
	SkipWhiteSpace();

	int result = 0;
	bool is_negative = false;
	if (content[cur_pos] == '-')
	{
		is_negative = true;
		cur_pos++;
	}

	while (cur_pos < file_size && content[cur_pos] >= '0' && content[cur_pos] <= '9')
	{
		result = result * 10 + (int)(content[cur_pos] - '0');
		cur_pos++;
	}
	cur_pos++;
	if (is_negative) 
		return -result;
	return result;
}

ele_type FileReader::ReadReal()
{
	SkipWhiteSpace();
	__int64 start = cur_pos;
	while (cur_pos < file_size && !IsWhiteSpace(content[cur_pos]))
		cur_pos++;
	content[cur_pos++] = '\0';
	return atof(&content[start]);
}

void FileReader::ReadString(char* str)
{
	SkipWhiteSpace();

	__int64 start = cur_pos;
	while (cur_pos < file_size && !IsWhiteSpace(content[cur_pos]))
		cur_pos++;

	memcpy(str, &content[start], cur_pos - start + 1);
	str[cur_pos - start] = '\0';
	cur_pos++;
}

int FileReader::ReadBinaryInt()
{
	int result;
	memcpy(&result, &content[cur_pos], sizeof(int));
	cur_pos += sizeof(int);
	return result;
}
	
float FileReader::ReadBinaryFloat()
{
	float result;
	memcpy(&result, &content[cur_pos], sizeof(float));
	cur_pos += sizeof(float);
	return result;
}

double FileReader::ReadBinaryDouble()
{
	double result;
	memcpy(&result, &content[cur_pos], sizeof(double));
	cur_pos += sizeof(double);
	return result;
}

void FileReader::CloseFile()
{
	file_size = 0;
	cur_pos = 0;
	free(content);
	content = 0;
}

ele_type Util::ComputeSRankCorr(int* rank0, int* rank1, int length)
{
	ele_type ans = 0;
	for(int i = 0; i < length; ++i)
		ans += (rank0[i] - rank1[i]) * (rank0[i] - rank1[i]);

	ans = 1.0 - 6 * ans / (length * (length * length - 1.0));

	return ans;
}

ele_type Util::MyLog(double x)
{
	return x < eps? MIN_LOG:log(x);
	//return log(x);
}

void Util::SoftMax(ele_type* in, ele_type* out, int size)
{
	ele_type sum = 0, max_v = in[0];
	for (int j = 1; j < size; ++j)
		max_v = (std::max)(max_v, in[j]);
	for (int j = 0; j < size; ++j)
		sum += exp(in[j] - max_v);
	for (int j = 0; j < size; ++j)
		out[j] = exp(in[j] - max_v) / sum;
}

ele_type Util::CompareRanks(int* rank0, int* rank1, int size, std::vector<int>& badIndexes, int badNum)
{
	ele_type loss = 0;

	if(badIndexes.size() != badNum)
		badIndexes.resize(badNum);

	std::vector<std::pair<int, int> > gaps;
	for(int i = 0; i < size; ++i)
		gaps.push_back(std::make_pair(abs(rank0[i] - rank1[i]), i));

	std::sort(gaps.begin(), gaps.end(), [](std::pair<int, int> a, std::pair<int, int> b) {
		return b.first < a.first;   
    });

	for(int i = 0; i < badNum; ++i)
	{
		badIndexes[i] = gaps[i].second;
		loss += (6.0 * gaps[i].first * gaps[i].first) / (size * (size * size - 1.0));
	}

	return loss;
}

bool Util::ReadWord(char *word, FILE *fin)
{
	int idx = 0;
	char ch;
	while (!feof(fin))
	{
		ch = fgetc(fin);
		if (ch == 13) continue;
		if ((ch == ' ') || (ch == '\t') || (ch == '\n')) 
		{
			if (idx > 0) 
			{
				if (ch == '\n') 
					ungetc(ch, fin);
				break;
			}
			if (ch == '\n') 
			{
				strcpy(word, (char *)"</s>");
				return true;
			}
			else continue;
		}
		word[idx++] = ch;
		if (idx >= MAX_WORD_LEN - 1) idx--;   // Truncate too long words
	}
	word[idx] = 0;
	return (bool)idx;
}