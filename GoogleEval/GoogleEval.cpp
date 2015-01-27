#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <algorithm>
#include "utility.h"
#include <vector>
#include <map>
#include <ctime>
#include <iostream>

#define MAX_WORD_LEN 150
#define sqr(x) ((x) * (x))

int vocab_size, embeding_size;
bool is_sparse = false;
char* embeding_file = NULL, *eval_file = NULL;
Trie trie;

EmbedingMatrix* embedings_full;
int emb_cnt;

std::vector<std::string> datasets_names;
std::vector<int> datasets_total;
std::vector<int> datasets_not_found;
std::vector<std::vector<TestSample> > datasets_full;
std::vector<int> eval_words_sense_idx;

std::map<int, int> word_evaluate_table;
int* mapped_word_evaluate_idx;
int* eval_word_start_idx_ininnerprod;

int evalset_vocab_size;
ele_type* inner_products;

struct wordInfo
{
	int idxInInput;
	
	int hsCodeLength;
	std::vector<int> hsNodesIdx;
	std::vector<int> hsBinCodes;

	int prototypeCnt;
	std::vector<ele_type> prototypePrior;

	char word[MAX_WORD_LEN];
};

wordInfo* words_info;

void ReadInputEmbeddings()
{
	printf("read input embeddings\t");
	FileReader reader;
	reader.OpenFile(embeding_file);

	vocab_size = reader.ReadInt();
	emb_cnt = reader.ReadInt();
	embeding_size = reader.ReadInt();

	embedings_full = new EmbedingMatrix(emb_cnt, embeding_size, false);
	words_info = new wordInfo[vocab_size];

	char word[MAX_WORD_LEN];
	
	int idx = 0;
	for(int i = 0; i < vocab_size; ++i)
	{
		reader.ReadString(word);
		
		trie.Insert(word);
		
		words_info[i].idxInInput = idx;
		strcpy(words_info[i].word, word);
		
		words_info[i].prototypeCnt = reader.ReadInt();

		for(int j = 0; j < words_info[i].prototypeCnt; ++j)
		{
			words_info[i].prototypePrior.push_back(reader.ReadBinaryFloat());

			for(int k = 0; k < embeding_size; ++k)
				embedings_full->dense(k, idx + j) = reader.ReadBinaryFloat();

			embedings_full->dense.col(idx + j).normalize();
		}
		idx += words_info[i].prototypeCnt;
	}
	reader.CloseFile();
}

void LoadEvaluationData()
{
	word_evaluate_table.clear();
	FILE* fid = fopen(eval_file, "r");
	char buf[MAX_WORD_LEN];
	fscanf(fid, "%s", buf);
	bool is_eof = false;
	while (!is_eof)
	{
		fscanf(fid, "%s", buf);
		datasets_names.push_back(std::string(buf));
		std::vector<TestSample> dataset;
		int total = 0, seen = 0;
		while (true)
		{
			if (fscanf(fid, "%s", buf) == EOF)
			{
				is_eof = true;
				break;
			}
			if (buf[0] == ':')
				break;

			bool valid = true;
			TestSample sample;
			sample.idxes[0] = trie.GetWordIndex(buf);
			valid &= sample.idxes[0] != -1;
			for (int i = 1; i < 4; ++i)
			{
				fscanf(fid, "%s", buf);
				sample.idxes[i] = trie.GetWordIndex(buf);
				valid &= sample.idxes[i] != -1;
			}
			if (valid)
			{
				seen++;
				dataset.push_back(sample);
				for (int i = 0; i < 4; ++i)
					if (word_evaluate_table.count(sample.idxes[i]) == 0)
					{
						int idx = word_evaluate_table.size();
						word_evaluate_table[sample.idxes[i]] = idx;
					}
			}
			total++;
		}
		datasets_full.push_back(dataset);
		datasets_not_found.push_back(total - seen);
		datasets_total.push_back(total);
	}

	evalset_vocab_size = word_evaluate_table.size();
	mapped_word_evaluate_idx = (int*) malloc(sizeof(int) * evalset_vocab_size);
	
	for (std::map<int, int>::iterator it = word_evaluate_table.begin(); it != word_evaluate_table.end(); ++it)
		mapped_word_evaluate_idx[it->second] = it->first;

	fclose(fid);

	eval_word_start_idx_ininnerprod = (int*) malloc(sizeof(int) * evalset_vocab_size);

	int word_idx;
	for(int i = 0 ; i < evalset_vocab_size; ++i)
	{
		word_idx =  mapped_word_evaluate_idx[i];
		eval_word_start_idx_ininnerprod[i] = eval_words_sense_idx.size();
		for(int j = 0; j < words_info[word_idx].prototypeCnt; ++j)
			eval_words_sense_idx.push_back(words_info[word_idx].idxInInput + j);
	}
}

int main(int argc, char* argv[])
{
	Eigen::initParallel();
	time_t start = clock();
	for (int i = 1; i < argc; ++i)
	{
		if (i == 1) embeding_file = argv[i];
		if (i == 2) eval_file = argv[i];
		if (i == 3) is_sparse = (bool)atoi(argv[i]);
	}
	
	ReadInputEmbeddings();

	LoadEvaluationData();

	inner_products = new ele_type[eval_words_sense_idx.size() * emb_cnt];

	if (is_sparse)
	{
		#pragma omp parallel for
		for (int i = 0; i < evalset_vocab_size; ++i)
		{
			embedings_full->GetSparseInnerProducts(mapped_word_evaluate_idx[i], i * vocab_size, inner_products);
		}
	} else
	{
		#pragma omp parallel for	
		for (int i = 0; i < eval_words_sense_idx.size(); ++i)
		{
			Matrix dots = embedings_full->dense.col(eval_words_sense_idx[i]).transpose() * embedings_full->dense;
			memcpy(inner_products + i * emb_cnt, dots.data(), sizeof(ele_type) * dots.cols());
		}
	}

	int total_samples = 0, total_seen = 0, total_correct = 0, max_db_size = 0;
	for (int d = 0; d < datasets_full.size(); ++d)
		if (datasets_full[d].size() > max_db_size)
			max_db_size = datasets_full[d].size();

	int* succ = (int*)malloc(max_db_size * sizeof(int));

	char result_file[100];
	sprintf(result_file, "%s.output.txt", embeding_file);
	FILE* fid = fopen(result_file, "w");
	for (int d = 0; d < datasets_full.size(); ++d)
	{
		memset(succ, 0, max_db_size * sizeof(int));
		#pragma omp parallel for
		for (int i = 0; i < datasets_full[d].size(); ++i)
		{
			ele_type similarity;
			TestSample* sample = &datasets_full[d][i];
			ele_type best_sim;
			int result_idx = -1;

			/*for (int j = 0; j < 4; ++j)
				sample->idxes[j] = word_evaluate_table[sample->idxes[j]];

			for (int j = 0; j < vocab_size; ++j)
			{
				if (j == mapped_word_evaluate_idx[sample->idxes[0]] || j == mapped_word_evaluate_idx[sample->idxes[1]] || j == mapped_word_evaluate_idx[sample->idxes[2]])
					continue;

				similarity = inner_products[sample->idxes[1] * vocab_size + j] - inner_products[sample->idxes[0] * vocab_size + j] + inner_products[sample->idxes[2] * vocab_size + j];
				if (result_idx < 0 || similarity > best_sim)
				{
					best_sim = similarity; 
					result_idx = j;
				}
			}*/

			int id0, id1, id2;
			int word_sense_id;
			for(int i0 = 0; i0 < words_info[sample->idxes[0]].prototypeCnt; ++i0)
			{
				id0 = eval_word_start_idx_ininnerprod[word_evaluate_table[sample->idxes[0]]] + i0;
				for(int i1 = 0; i1 < words_info[sample->idxes[1]].prototypeCnt; ++i1)
				{
					id1 = eval_word_start_idx_ininnerprod[word_evaluate_table[sample->idxes[1]]] + i1;
					for(int i2 = 0; i2 < words_info[sample->idxes[2]].prototypeCnt; ++i2)
					{
						id2 = eval_word_start_idx_ininnerprod[word_evaluate_table[sample->idxes[2]]] + i2;
						for(int w = 0; w < vocab_size; ++w)
						{
							if (w == sample->idxes[0] || w == sample->idxes[1] || w == sample->idxes[2]) continue;

							for(int i3 = 0; i3 < words_info[w].prototypeCnt; ++i3)
							{
								word_sense_id = words_info[w].idxInInput + i3;
								similarity = inner_products[id1 * emb_cnt + word_sense_id] - inner_products[id0 * emb_cnt + word_sense_id] + inner_products[id2 * emb_cnt + word_sense_id];
								if(result_idx < 0 || similarity > best_sim)
								{
									best_sim = similarity;
									result_idx = w;
								}
							}
						}
					}
				}
			}
			if (result_idx == sample->idxes[3])
				succ[i] = 1;
		}

		int acc = 0;
		for (int i = 0; i < datasets_full[d].size(); ++i)
			acc += succ[i];
		fprintf(fid, "%s, total %d, not found %d, correct %d, accuracy %.2f %%\n", datasets_names[d].c_str(), datasets_total[d], datasets_not_found[d], acc, 100 * (acc + 0.0) / (datasets_total[d]));
		printf("%s, total %d, not found %d, correct %d, accuracy %.2f %%\n", datasets_names[d].c_str(), datasets_total[d], datasets_not_found[d], acc, 100 * (acc + 0.0) / (datasets_total[d]));
		total_samples += datasets_total[d];
		total_seen += datasets_total[d] - datasets_not_found[d];
		total_correct += acc;
	}

	fprintf(fid, "Total %d, TotalEvaluated %d, Total Correct %d, Total Accuracy %.2f %%\n", total_samples, total_seen, total_correct, 100.0 * total_correct / total_samples);
	printf("Total %d, TotalEvaluated %d, Total Correct %d, Total Accuracy %.2f %%\n", total_samples, total_seen, total_correct, 100.0 * total_correct / total_samples);
	fclose(fid);

	time_t end = clock();
	printf("Time elapsed %.4f seconds\n", (double)(end - start) / CLOCKS_PER_SEC);
	return 0;
}