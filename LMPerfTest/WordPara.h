#include <vector>

typedef float ele_type;

const int MAX_WORD_LEN = 100, MAX_CTX_CNT = 100, MAX_WORD_IN_SEN = 150, MAX_FILENAME_LEN = 1024, MAX_SENSE_CNT = 50, MAX_HF_CODE_LEN = 1000, MAX_SENTENCE_LEN = 1000;

struct wordInfo
{
	int idxInInput;
	
	int hsCodeLength;
	std::vector<int> hsNodesIdx;
	std::vector<int> hsBinCodes;

	int prototypeCnt;
	std::vector<ele_type> prototypePrior;
	std::vector<ele_type> eqlogv;
	std::vector<ele_type> eqlog1_v;
	std::vector<ele_type> eta_sigmas;
	std::vector<ele_type> eta_norms;

	ele_type in_llbound;
	ele_type out_llbound;
	ele_type sigma_V;
	ele_type alpha;

	char word[MAX_WORD_LEN];
};

