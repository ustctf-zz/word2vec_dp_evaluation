// LMPerfTest.cpp : Defines the entry point for the console application.
//
#include <stdio.h>
#include <string.h>
#include "utility.h"
#include <iostream>

char input_emb_filename[MAX_FILENAME_LEN], output_emb_filename[MAX_FILENAME_LEN], hstree_filename[MAX_FILENAME_LEN];
char test_filename[MAX_FILENAME_LEN], sw_filename[MAX_FILENAME_LEN];

int vocab_size, input_emb_size, emb_dim;
long long word_count, last_word_count, word_count_actual, test_total_words_cnt;

bool is_second_approx, is_hs = true, is_stop_word = true, is_npb = true;
FILE* test_fin;

wordInfo* words_info;
EmbedingMatrix* inputEmbedings, *outputEmbeddings;
Trie trie, sw_trie;

ele_type* out_emb_norm_table, *out_emb_sigma, *sigma_U;

 void ParseArgs(int argc, char* argv[])
{
	for (int i = 1; i < argc; i += 2)
	{
		if (strcmp(argv[i], "-emb") == 0)		strcpy(input_emb_filename, argv[i + 1]);
		if (strcmp(argv[i], "-test") == 0)		strcpy(test_filename, argv[i + 1]);
		if (strcmp(argv[i], "-hsfile") == 0)	strcpy(hstree_filename, argv[i + 1]);
		if (strcmp(argv[i], "-stopwords") == 0) is_stop_word = (bool)atoi(argv[i + 1]);
		if (strcmp(argv[i], "-is_npb") == 0)	is_npb = (bool)atoi(argv[i + 1]);
		if (strcmp(argv[i], "-sw_file") == 0)   strcpy(sw_filename, argv[i + 1]);
		if (strcmp(argv[i], "-out_emb") == 0)	strcpy(output_emb_filename, argv[i + 1]);
	}
}

void ReadStopWords()
{
	FILE* fid = fopen(sw_filename,"r");

	char word[MAX_WORD_LEN];
	while(fscanf(fid,"%s", word) != EOF)
		sw_trie.Insert(word);
	
	fclose(fid);
}

void ReadInputEmbeddings_EM()
{
	printf("read input embeddings\t");
	FileReader reader;
	reader.OpenFile(input_emb_filename);

	vocab_size = reader.ReadInt();
	input_emb_size = reader.ReadInt();
	emb_dim = reader.ReadInt();

	inputEmbedings = new EmbedingMatrix(input_emb_size, emb_dim, false);
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

			for(int k = 0; k < emb_dim; ++k)
				inputEmbedings->dense(k, idx + j) = reader.ReadBinaryFloat();

			inputEmbedings->dense.col(idx + j).normalize();
		}
		idx += words_info[i].prototypeCnt;
	}
	reader.CloseFile();
}

void ReadOutputEmbeddings_EM()
{
	printf("Read output embeddings\t");

	FileReader reader;
	reader.OpenFile(output_emb_filename);

	int outvocab_size = reader.ReadInt();
	emb_dim = reader.ReadInt();

	outputEmbeddings = new EmbedingMatrix(outvocab_size, emb_dim, false);

	if(is_hs)
	{
		for(int i = 0;i < outvocab_size; ++i)
		{
			for(int j = 0; j < emb_dim; ++j)
				outputEmbeddings->dense(j, i) = reader.ReadBinaryFloat();
		}
	}
	

	reader.CloseFile();
}

void ReadInputEmbeddings_New()
{
	printf("read input embeddings\t");
	FileReader reader;
	reader.OpenFile(input_emb_filename);

	vocab_size = reader.ReadInt();
	input_emb_size = reader.ReadInt();
	emb_dim = reader.ReadInt();
	is_second_approx = reader.ReadInt();

	inputEmbedings = new EmbedingMatrix(input_emb_size, emb_dim, false);
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
		words_info[i].alpha = reader.ReadBinaryFloat();
		words_info[i].in_llbound = reader.ReadBinaryFloat();
		words_info[i].out_llbound = reader.ReadBinaryFloat();

		for(int j = 0; j < words_info[i].prototypeCnt; ++j)
		{
			words_info[i].prototypePrior.push_back(reader.ReadBinaryFloat());

			words_info[i].sigma_V = reader.ReadBinaryFloat();

			words_info[i].eqlogv.push_back(reader.ReadBinaryFloat());
			words_info[i].eqlog1_v.push_back(reader.ReadBinaryFloat());

			words_info[i].eta_sigmas.push_back(reader.ReadBinaryFloat());
			words_info[i].eta_norms.push_back(reader.ReadBinaryFloat());

			for(int k = 0; k < emb_dim; ++k)
				inputEmbedings->dense(k, idx + j) = reader.ReadBinaryFloat();
		}
		idx += words_info[i].prototypeCnt;
	}
	reader.CloseFile();
}

void ReadOutputEmbeddings_New()
{
	printf("Read output embeddings\t");

	FileReader reader;
	reader.OpenFile(output_emb_filename);

	int outvocab_size = reader.ReadInt();
	emb_dim = reader.ReadInt();

	outputEmbeddings = new EmbedingMatrix(outvocab_size, emb_dim, false);
	out_emb_norm_table = (ele_type*) malloc(sizeof(ele_type) * outvocab_size);
	out_emb_sigma = (ele_type*) malloc(sizeof(ele_type) * outvocab_size);
	sigma_U = (ele_type*) malloc(sizeof(ele_type) * outvocab_size);

	if(is_hs)
	{
		/*for(int i = 0; i < outvocab_size; ++i)
		{
			sigma_U[i] = reader.ReadBinaryFloat();
			out_emb_norm_table[i] = reader.ReadBinaryFloat();
			out_emb_sigma[i] = reader.ReadBinaryFloat();

			for(int j = 0; j < emb_dim; ++j)
				outputEmbeddings->dense(j, i) = reader.ReadBinaryFloat();
		}*/
		ele_type sigma_u = reader.ReadBinaryFloat();

		for(int i = 0; i < outvocab_size; ++i)
			sigma_U[i] = sigma_u;
		for(int i = 0; i < outvocab_size; ++i)
			out_emb_norm_table[i] = reader.ReadBinaryFloat();

		for(int i = 0; i < outvocab_size; ++i)
			out_emb_sigma[i] = reader.ReadBinaryFloat();

		for(int i = 0; i < outvocab_size; ++i)
			for(int j = 0; j < emb_dim; ++j)
				outputEmbeddings->dense(j, i) = reader.ReadBinaryFloat();
	}
	else
	{
		char word[MAX_WORD_LEN];
		for(int i = 0; i < outvocab_size; ++i)
		{
			reader.ReadString(word);
			int wordIdx = trie.GetWordIndex(word);
			for(int j = 0;j < emb_dim; ++j)
				outputEmbeddings->dense(j, wordIdx) = reader.ReadBinaryFloat();
		}
	}

	reader.CloseFile();
}

void ReadHsStructures()
{
	printf("read hs structures\t");
	FileReader reader;
	reader.OpenFile(hstree_filename);

	vocab_size = reader.ReadInt();

	char word[MAX_WORD_LEN];
	int wordIdx, codeLen;
	for(int i = 0; i < vocab_size; ++i)
	{
		reader.ReadString(word);
		wordIdx = trie.GetWordIndex(word);

		codeLen = reader.ReadInt();

		for(int j = 0; j < codeLen; ++j)
			words_info[wordIdx].hsBinCodes.push_back(reader.ReadInt());

		for(int j = 0; j < codeLen; ++j)
			words_info[wordIdx].hsNodesIdx.push_back(reader.ReadInt());

		words_info[wordIdx].hsCodeLength = codeLen;
	}
}

void VariationalEStep(int inIdx, int outIdx, ele_type* phi, ele_type* EqLogp)
{
	int T = words_info[inIdx].prototypeCnt;
	memset(phi, 0, sizeof(ele_type) * T);
	
	for(int t = 0; t < T; ++t)
	{
		EqLogp[t] = Util::MyLog(words_info[inIdx].prototypePrior[t]);
		for(int d = 0; d < words_info[outIdx].hsCodeLength; ++d)
		{
			ele_type innerProduct = inputEmbedings->dense.col(words_info[inIdx].idxInInput + t).dot(outputEmbeddings->dense.col(words_info[outIdx].hsNodesIdx[d]));
			if(words_info[outIdx].hsBinCodes[d] == 0)//If it is 0, then use sigmoid(x) = 1 / (1 + exp(-x))
					innerProduct = -innerProduct;
			if(!is_second_approx)
				EqLogp[t] -= log(1 + exp(innerProduct));
			else
			{
				//TODO: ADD SECOND APPROXIMATION
			}
		}
	}
	Util::SoftMax(EqLogp, phi, T);
	for(int t = 0; t < T; ++t)
		EqLogp[t] *= phi[t];
}

ele_type ComputeLogProb_EM(int inIdx, int outIdx)
{
	ele_type ans = 0, log_prob;
	
	int T = words_info[inIdx].prototypeCnt;

	if(is_hs)
	{
		for(int t = 0; t < T; ++t)
		{
			log_prob = Util::MyLog(words_info[inIdx].prototypePrior[t]);

			for(int i = 0; i < words_info[outIdx].hsCodeLength; ++i)
			{
				ele_type innerProduct = inputEmbedings->dense.col(words_info[inIdx].idxInInput + t).dot(outputEmbeddings->dense.col(words_info[outIdx].hsNodesIdx[i]));
				if(words_info[outIdx].hsBinCodes[i] == 0)//If it is 0, then using sigmoid(x) = 1 / (1 + exp(-x))
					innerProduct = -innerProduct;
				log_prob -= log(1 + exp(innerProduct));
			}
			ans += exp(log_prob);
		}
	}
	
	return log(ans);
}

//Computer logProb(w_{outIdx}|w_{inIdx})
ele_type ComputeLogProb_New(int inIdx, int outIdx)
{
	ele_type prob = 0, ans = 0;
	
	if(words_info[inIdx].prototypeCnt == 1)
	{
		if(is_hs)
		{
			for(int i = 0; i < words_info[outIdx].hsCodeLength; ++i)
			{
				ele_type innerProduct = inputEmbedings->dense.col(words_info[inIdx].idxInInput).dot(outputEmbeddings->dense.col(words_info[outIdx].hsNodesIdx[i]));
				if(words_info[outIdx].hsBinCodes[i] == 0)//If it is 0, then use sigmoid(x) = 1 / (1 + exp(-x))
					innerProduct = -innerProduct;
				prob -= log(1 + exp(innerProduct));
			}
		}
		else
			prob = Util::MyLog(outputEmbeddings->dense.col(outIdx).dot(inputEmbedings->dense.col(words_info[inIdx].idxInInput)));

		return prob;
	}

	ans = (words_info[inIdx].in_llbound + words_info[outIdx].out_llbound);

	ele_type phi[MAX_SENSE_CNT], Eqlogpw[MAX_SENSE_CNT];
	VariationalEStep(inIdx, outIdx, phi, Eqlogpw);

	ele_type phi_sum = 0;
	for(int t = 0; t < words_info[inIdx].prototypeCnt; ++t)
		phi_sum += phi[t];

	for(int sense_idx = 0 ; sense_idx < words_info[inIdx].prototypeCnt; ++sense_idx)
	{
		ans += Eqlogpw[sense_idx];
		phi_sum -= phi[sense_idx];
		ans += (phi[sense_idx] * words_info[inIdx].eqlogv[sense_idx] + phi_sum * words_info[inIdx].eqlog1_v[sense_idx]);

		ans -= phi[sense_idx] < eps? 0 : phi[sense_idx] * log(phi[sense_idx]);
	}
	return ans;
}

int GetSentence(int* sentence, FILE* fin)
{
	int length = 0, word_idx;
	char word[MAX_WORD_LEN];
	while (1)
	{
		if (!Util::ReadWord(word, fin))
			break;

		word_count++;
		word_idx = trie.GetWordIndex(word);
		if (word_idx == -1)
			continue;

		if (is_stop_word && sw_trie.GetWordIndex(word) != -1)
			continue;
		sentence[length++] = word_idx;
		if (length >= MAX_SENTENCE_LEN)
			break;
	}
	if (word_count - last_word_count > 10000) 
	{
		word_count_actual += word_count - last_word_count;
		last_word_count = word_count;
		float progress = word_count_actual / (ele_type)(test_total_words_cnt + 1);
		printf("%c  Progress: %.2f%%  ", 13, progress * 100);
		fflush(stdout);
	}
	return length;
}

ele_type ComputePerplexity()
{
	int sentence[1000 + 2], sentence_length;
	long long pair_cnt = 0;
	ele_type perplexity = 0;

	ele_type pair_ppl;
	while (1)
	{
		sentence_length = GetSentence(sentence, test_fin);
		if (sentence_length == 0)
			break;
		for (int sentence_position = 0; sentence_position < sentence_length - 1; ++sentence_position)
		{
			pair_ppl = is_npb? ComputeLogProb_EM(sentence[sentence_position], sentence[sentence_position + 1]) : ComputeLogProb_New(sentence[sentence_position], sentence[sentence_position + 1]);
			perplexity += pair_ppl;
			pair_cnt++;
		}
	}
	return exp(-perplexity / pair_cnt);
}

int main(int argc, char* argv[])
{
	ParseArgs(argc, argv);

	if(!is_npb)
	{
		ReadInputEmbeddings_New();
		ReadOutputEmbeddings_New();
	}
	else
	{
		ReadInputEmbeddings_EM();
		ReadOutputEmbeddings_EM();
	}

	ReadHsStructures();

	char word[MAX_WORD_LEN];
	test_fin = fopen(test_filename, "r");
	test_total_words_cnt = 0;
	while (Util::ReadWord(word, test_fin))
		test_total_words_cnt++;
	fclose(test_fin);

	word_count = 0;
	last_word_count = 0;

	printf("%I64d\n", test_total_words_cnt);
	
	test_fin = fopen(test_filename, "r");
	ele_type ppl = ComputePerplexity();

	printf("The perplexity is %.3f\n", ppl);
	fclose(test_fin);

	/*int a, b ,c,d,e ;
	char word[100];
	float x[5],y,z;
	FILE* fid = fopen(input_emb_filename, "rb");
	fscanf(fid, "%d %d %d %d", &a, &b, &c, &d);
	fgetc(fid);
	fscanf(fid, "%s %d", word, &e);

	fgetc(fid);
	fread(&x, sizeof(ele_type), 3, fid);
	fread(&y, sizeof(ele_type), 1, fid);
	fread(&z, sizeof(ele_type), 1, fid);*/

	return 0;
}