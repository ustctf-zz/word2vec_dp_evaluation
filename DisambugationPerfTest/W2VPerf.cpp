// W2VPerf.cpp : Defines the entry point for the console application.
//

#include "utility.h"
#include <stdio.h>
#include <algorithm>
#include <string>

//Check word2vec's performance on WordSim353 dataset and EH's new dataset

char ws353file[20],ehnewfile[20],embeding_file[100];

int embedingsize, vocab_size;
const int datasize0 = 353, datasize1 = 2003;
Trie trie;
ele_type* wordembeddings;

struct wordpair{
	char word0[100],word1[100];
	int id,wordid0,wordid1;
	double simscore;
	double w2vscore;
};

int compare_orig(const void* wp0,const void* wp1)
{
	double score0=((wordpair*)wp0)->simscore,score1=((wordpair*)wp1)->simscore;
	return score0<score1?1:-1;
}

int compair_w2v(const void* wp0,const void* wp1)
{
	double score0=((wordpair*)wp0)->w2vscore,score1=((wordpair*)wp1)->w2vscore;
	return score0<score1?1:-1;
}

int bad_cnt0 = 0, bad_cnt1 = 0;

wordpair data353[400],ehdata[2005];

int special_idx = 0;
void ReadTestData()
{
	char buf[2000];
	FILE* fid;
	errno_t erno = fopen_s(&fid, ws353file, "r");
	fgets(buf, 100, fid);

	for (int i = 0; i < datasize0; ++i)
	{
		fgets(buf, 200, fid);
		strcpy(data353[i].word0, strtok(buf, ","));
		strcpy(data353[i].word1, strtok(NULL, ","));
		data353[i].simscore = atof(strtok(NULL, ","));
		data353[i].id = i;

		data353[i].wordid0 = trie.GetWordIndex(data353[i].word0);
		data353[i].wordid1 = trie.GetWordIndex(data353[i].word1);
	}

	fclose(fid);

	fopen_s(&fid, ehnewfile, "r");

	char* token, *p;
	int idx = 0;
	while (fgets(buf, 2000, fid) != NULL)
	{
		token = strtok_s(buf, "\t", &p);
		for (int i = 0; i < 8; ++i)
		{
			token = strtok_s(NULL, "\t", &p);
			if (i == 0) strcpy(ehdata[idx].word0, token);
			if (i == 2) strcpy(ehdata[idx].word1, token);
			if (i == 6) ehdata[idx].simscore = atof(token);
		}
		ehdata[idx].id = idx;
		//if (strcmp(ehdata[idx].word0, "rock") == 0 && strcmp(ehdata[idx].word1, "jazz") == 0)
		if (strcmp(ehdata[idx].word0, "star") == 0 && strcmp(ehdata[idx].word1, "star") == 0)
			special_idx = idx;

		ehdata[idx].wordid0 = trie.GetWordIndex(ehdata[idx].word0);
		ehdata[idx].wordid1 = trie.GetWordIndex(ehdata[idx].word1);
		idx++;
	}

	fclose(fid);
}

void ReadEmbeddingData()
{
	FileReader reader;
	reader.OpenFile(embeding_file);

	int vocab_size = reader.ReadInt();

	embedingsize = reader.ReadInt();
	wordembeddings = (ele_type*)malloc(sizeof(ele_type)*embedingsize*vocab_size);

	char word[105];
	for (int i = 0; i < vocab_size; ++i)
	{
		reader.ReadString(word);

		trie.Insert(word);

		ele_type norm = 0;
		for (int j = 0; j < embedingsize; ++j)
		{
			float tmp = reader.ReadBinaryFloat();
			norm += tmp*tmp;
			wordembeddings[i*embedingsize + j] = tmp;

		}
		norm = sqrt(norm);
		for (int j = 0; j < embedingsize; ++j)
			wordembeddings[i*embedingsize + j] /= norm;

	}
	reader.CloseFile();
}

ele_type ComputeSRankCorr(int* rank0, int* rank1, int length)
{
	ele_type ans = 0;
	for (int i = 0; i < length; ++i)
		ans += (rank0[i] - rank1[i])*(rank0[i] - rank1[i]);

	ans = 1.0 - 6 * ans / (length*(length*length - 1.0));

	return ans;
}

void ComputeW2VScore()
{
	for (int i = 0; i < datasize0; ++i)
	{
		data353[i].w2vscore = 0;
		if (data353[i].wordid0 == -1 || data353[i].wordid1 == -1)
		{
			bad_cnt0++;
			continue;
		}
		for (int j = 0; j < embedingsize; ++j)
			data353[i].w2vscore += wordembeddings[data353[i].wordid0 * embedingsize + j] * wordembeddings[data353[i].wordid1 * embedingsize + j];
	}

	for (int i = 0; i < datasize1; ++i)
	{
		ehdata[i].w2vscore = 0;

		if (ehdata[i].wordid0 == -1 || ehdata[i].wordid1 == -1)
		{
			bad_cnt1++;
			continue;
		}
		for (int j = 0; j < embedingsize; ++j)
			ehdata[i].w2vscore += wordembeddings[ehdata[i].wordid0*embedingsize + j] * wordembeddings[ehdata[i].wordid1*embedingsize + j];
	}
	printf("The score of special: %.4f\n", ehdata[special_idx].w2vscore);

}

int main(int argc, char* argv[])
{
	if (argc != 4)
	{
		printf("Invalide arg numbers!\n");
		return -1;
	}

	int j;
	for (j = 0; argv[1][j] != '\0'; ++j)
		ws353file[j] = (char)(argv[1][j]);
	ws353file[j] = '\0';

	for (j = 0; argv[2][j] != '\0'; ++j)
		ehnewfile[j] = (char)(argv[2][j]);
	ehnewfile[j] = '\0';

	for (j = 0; argv[3][j] != '\0'; ++j)
		embeding_file[j] = (char)(argv[3][j]);
	embeding_file[j] = '\0';

	ReadEmbeddingData();
	ReadTestData();

	ComputeW2VScore();

	int rankgold353[datasize0], rankgoldeh[datasize1];
	int rank353[datasize0], rankeh[datasize1];

	std::qsort(data353, datasize0, sizeof(wordpair), compare_orig);
	for (int i = 0; i < datasize0; ++i)
		rankgold353[data353[i].id] = i;

	std::qsort(data353, datasize0, sizeof(wordpair), compair_w2v);

	for (int i = 0; i < datasize0; ++i)
		rank353[data353[i].id] = i;

	std::qsort(ehdata, datasize1, sizeof(wordpair), compare_orig);

	for (int i = 0; i < datasize1; ++i)
		rankgoldeh[ehdata[i].id] = i;
	
	std::qsort(ehdata, datasize1, sizeof(wordpair), compair_w2v);
	for (int i = 0; i < datasize1; ++i)
		rankeh[ehdata[i].id] = i;

	printf("gold rank : %d, w2v rank: %d\n", rankgoldeh[special_idx], rankeh[special_idx]);

	ele_type pcorrW2V0 = ComputeSRankCorr(rankgold353, rank353, datasize0);
	ele_type pcorrW2V1 = ComputeSRankCorr(rankgoldeh, rankeh, datasize1);

	free(wordembeddings);

	printf("%f %f\n\n", pcorrW2V0, pcorrW2V1);

	return 0;
}

