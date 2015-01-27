#pragma once

#include <cstring>
#include <climits>
#include <vector>
#include <Eigen\Dense>
#include <map>
#include "WordPara.h"

typedef Eigen::Matrix<ele_type, Eigen::Dynamic, Eigen::Dynamic> Matrix;

#define MIN_LOG -10

const ele_type eps = 1e-6;

const int MAX_WINDOW_SIZE = 20;

struct FeatPair
{
	int idx;
	ele_type value;

	bool operator < (const FeatPair& other) const
	{
		return idx < other.idx;
	}

	FeatPair(int _idx = -1, ele_type _value = -1)
	{
		idx = _idx;
		value = _value;
	}
};

struct TestSample
{
	int idxes[4];
};

struct EmbedingMatrix
{
	EmbedingMatrix(int _word_table_size, int _embeding_size, bool _is_sparse);

	void Destroy();

	void GetSparseInnerProducts(int word_idx, int base, ele_type* innter_products);

	int word_table_size; 
	int embeding_size;
	bool is_sparse;
	FeatPair** sparse;
	Matrix dense;
	std::vector<std::vector<FeatPair> > feat_table;
};

class FileReader
{
public:
	FileReader();
	void OpenFile(const char* filename);
	void CloseFile();
	int ReadInt();
	int ReadBinaryInt();
	float ReadBinaryFloat();
	double ReadBinaryDouble();
	ele_type ReadReal();
	void ReadString(char* str);

private:
	void SkipWhiteSpace();
	char* content;
	__int64 file_size, cur_pos;
};

inline bool IsWhiteSpace(char &ch)
{
	return (ch == '\t' || ch == '\r' || ch == '\n' || ch == ' ');
}

class TreeNode
{
public:
	int word_idx;
	TreeNode* children[128];
	TreeNode()
	{
		word_idx = -1;
		memset(children, 0, sizeof(TreeNode*) * 128);
	}
};

struct Trie
{
	Trie()
	{
		root = new TreeNode();
		word_cnt = 0;
	}

	void Insert(char* word)
	{
		TreeNode* node = root;
		int pos = 0, branch;
		while (word[pos])
		{
			branch = (word[pos] >= 'A' && word[pos] <= 'Z') ? word[pos] - 'A' + 'a' : word[pos];
			if (!node->children[branch])
				node->children[branch] = new TreeNode();
			node = node->children[branch];
			pos++;
		}

		if (node->word_idx == -1)
			node->word_idx = word_cnt++;
	}

	int GetWordIndex(char* word)
	{
		TreeNode* node = root;
		int pos = 0, branch;
		while (word[pos])
		{
			branch = (word[pos] >= 'A' && word[pos] <= 'Z') ? word[pos] - 'A' + 'a' : word[pos];
			if (branch < 0 || branch > 127 || !node->children[branch]) 
				return -1;
			node = node->children[branch];
			pos++;
		}
		return node->word_idx;
	}

	
	int word_cnt;
private:
	TreeNode* root;
};

class Util
{
public:
	static ele_type ComputeSRankCorr(int*, int*, int);
	static ele_type MyLog(double x);
	static void SoftMax(ele_type* in, ele_type* out, int length);
	static ele_type CompareRanks(int* rank0, int* rank1, int size, std::vector<int>& badIndexes, int badNum);
	static bool ReadWord(char *word, FILE *fin);
};


