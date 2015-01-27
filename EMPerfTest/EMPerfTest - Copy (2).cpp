// EMPerfTest.cpp : Defines the entry point for the console application.
//

//#include "stdafx.h"

//argv[1]:combined.csv
//argv[2]:ratings.txt
//argv[3]:input embeddings file;
//argv[4]:output embeddings file, either hs or word embedding
//argv[5]:hs index file
//argv[6]:-s
//argv[7]:sim type:0,average;1,weighted;2,max
//argv[8]:-sw
//argv[9]:whether using stop words
//argv[10]:stop words file
//argv[11]:-tfidf
//argv[12]:whether using tf*idf weighting in test
//argv[13]:whether logging bad cases info
//argv[14]:bad cases number
//argv[15]:bad cases logging file

#include "utility.h"
#include <ctime>
#include <algorithm>
#include <random>
#include <io.h>
#include <set>

enum simType
{ //Similarity type used to define sim score without Context
	Uniform,
	Weighted,
	Max
};
int vocabSize, embedDim, inputEmbedSize, outEmbedSize;
const int MAX_WORD_LEN = 105, MAX_CTX_CNT = 100, MAX_WORD_IN_SEN = 150, MAX_FILENAME_LEN = 1024, MAX_SENSE_CNT = 50, MAX_HF_CODE_LEN = 1000;
const int DATA0_SIZE = 353, DATA1_SIZE = 2003;
const int MAX_WINDOW_SIZE = 20;

char data353File[MAX_FILENAME_LEN], ehdataFile[MAX_FILENAME_LEN], inputEmbedFile[MAX_FILENAME_LEN], outEmbedFile[MAX_FILENAME_LEN], hstreeFile[MAX_FILENAME_LEN], swFile[MAX_FILENAME_LEN], badCaseFile[MAX_FILENAME_LEN];
int isHs = 1, isStopWords = 0, isTfIdf = 0;
simType mysimType;

std::set<long long> wordsInTest;

int badCaseNum = 0;

int rankGoldWS[DATA0_SIZE], rankGoldEH[DATA1_SIZE];
int rankWS[DATA0_SIZE], rankEH[DATA1_SIZE];
int bestRankEh[DATA1_SIZE];
ele_type bestEhSimScores[DATA1_SIZE];

bool ParseCmdArgs(int argc, char* argv[])
{
	strcpy(data353File, argv[1]);
	strcpy(ehdataFile, argv[2]);
	strcpy(inputEmbedFile, argv[3]);
	strcpy(outEmbedFile, argv[4]);
	strcpy(hstreeFile, argv[5]);

	if(_access(hstreeFile,0) == -1)
		isHs = 0;

	char paraType[20];
	for(int i = 6; i < argc; i += 2)
	{
		if(argv[i][0] == '-')
			strcpy(paraType, argv[i] + 1);

		if(strcmp(paraType, "s") == 0)
		{
			switch (atoi(argv[i + 1]))
			{
				case 0:
					mysimType = Uniform;
					printf("sim:Uniform, isHS:%d\n", isHs);
					break;
				case 1:
					mysimType = Weighted;
					printf("sim:Weighted, isHS:%d\n", isHs);
					break;
				case 2:
					mysimType = Max;
					printf("sim:Max, isHS:%d\n", isHs);
					break;
				default:
					return false;
			}
		}
		else if(strcmp(paraType, "sw") == 0)
		{
			isStopWords = atoi(argv[i + 1]);
			if(isStopWords)
			{
				strcpy(swFile, argv[i + 2]);
				i++;
			}
		}
		else if(strcmp(paraType, "tfidf") == 0)
			isTfIdf = atoi(argv[i + 1]);

		else if(strcmp(paraType, "d") == 0)
		{
			badCaseNum = atoi(argv[i + 1]);
			strcpy(badCaseFile, argv[i + 2]);
			i++;
		}
		else
		{
			printf("format error!\n");
			return false;
		}
	}

	return true;
}

Trie trie;
Trie swTrie;

int badCnt = 0;

EmbedingMatrix* inputEmbedings, *outputEmbeddings;

struct wordInfo
{
	int idxInInput;
	
	int hsCodeLength;
	std::vector<int> hsNodesIdx;
	std::vector<int> hsBinCodes;

	int prototypeCnt;
	std::vector<ele_type> prototypePrior;

	char word[MAX_WORD_LEN];

	ele_type* tfidfs;
	ele_type tf, idf;

	wordInfo()
	{
		if(isTfIdf)
		{
			tfidfs = (ele_type*)calloc(2 * DATA1_SIZE, sizeof(ele_type));
			idf = 0;
			tf = 0;
		}
	}
};

wordInfo* wordsInfo;

struct wordPair
{
	char word0[MAX_WORD_LEN],word1[MAX_WORD_LEN];

	int idx0, idx1;//word idx in wordsInfo

	int ctx0[MAX_CTX_CNT], ctx1[MAX_CTX_CNT];

	ele_type simScoreWithoutCtx;
	ele_type simScoreWithCtx;

	ele_type goldSimScore;

	int id;
	int ctxSize0,ctxSize1;
};

void ReadStopWords()
{
	FILE* fid = fopen(swFile,"r");

	char word[MAX_WORD_LEN];
	while(fscanf(fid,"%s", word) != EOF)
		swTrie.Insert(word);
	
	fclose(fid);
}

int CmpWithGold(const void* wp0, const void* wp1) {return ((wordPair*)(wp0))->goldSimScore<((wordPair*)(wp1))->goldSimScore?1:-1;}

int CmpWithOutCtx(const void* wp0,const void* wp1){ return ((wordPair*)(wp0))->simScoreWithoutCtx <((wordPair*)(wp1))->simScoreWithoutCtx?1:-1;}

int CmpWithCtx(const void* wp0,const void* wp1){ return ((wordPair*)(wp0))->simScoreWithCtx <((wordPair*)(wp1))->simScoreWithCtx?1:-1;}

int CmpWithIdx(const void* wp0, const void* wp1){ return ((wordPair*)(wp0))->id > ((wordPair*)(wp1))->id? 1: -1;}

wordPair data353[DATA0_SIZE], ehdata[DATA1_SIZE];

int GetWordsInSen(const char* sentence, int* words)
{
	int idx = 0, widx = 0, word_idx;
	bool isInWord = false;
	char c, word[MAX_WORD_LEN];
	for(int i = 0; (c = sentence[i]) != '\0'; ++i)
	{
		if(c >= 'a' && c <= 'z' || c >= 'A' && c <= 'Z')
		{
			isInWord = true;
			word[widx++] = tolower(c);
		}
		else if(c != '<')
		{
			word[widx] = '\0';
			if(isInWord && (!isStopWords || swTrie.GetWordIndex(word) == -1))
			{
				word_idx = trie.GetWordIndex(word);
				if(word_idx != -1)
					words[idx++] = word_idx;
			}
			isInWord = false;
			widx = 0;
		}
		else
			while(sentence[i]!='>') i++;
	}
	if(isInWord)
	{
		word[widx] = '\0';
		if(!isStopWords || swTrie.GetWordIndex(word) == -1)
		{
			word_idx = trie.GetWordIndex(word);
			if(word_idx != -1)
				words[idx++] = word_idx;
		}
	}
	return idx;
}

void GetCtxWords(const int* sentence, const int senLen, const int wordIdx, int* ctx, int& ctxSize)
{
	int wordPos = -1;
	for(int i = 0; i < senLen; ++i)
	{
		if(sentence[i] == wordIdx)
		{
			wordPos = i;
			break;
		}
	}

	ctxSize = 0;

	int p, loc;
	for(int a = 0; a < 2 * MAX_WINDOW_SIZE + 1; ++a)
	{
		p = wordPos - MAX_WINDOW_SIZE + a;
		if(a == MAX_WINDOW_SIZE || p < 0 || p >= senLen || sentence[p] == -1)
			continue;
		
		loc = 2 * abs(p - wordPos - 1) + (a < MAX_WINDOW_SIZE? 0 : 1);
		ctx[loc] = sentence[p];
		ctxSize++;
	}
}

void CalTfScore(int senLen, int* sentence, int docid)
{
	int word;
	for(int i = 0; i < senLen; ++i)
	{
		word = sentence[i];
		if(word == -1)
			continue;

		wordsInTest.insert(word);
		if(wordsInfo[word].tf == 0)
			wordsInfo[word].idf += 1;
		wordsInfo[word].tf += 1;
	}

	for(int i = 0; i < senLen; ++i)
	{
		word = sentence[i];
		if(word == -1)
			continue;
		wordsInfo[sentence[i]].tfidfs[docid] = wordsInfo[sentence[i]].tf / senLen;
	}

	for(int i = 0; i < senLen; ++i)
	{
		word = sentence[i];
		if(word == -1)
			continue;
		wordsInfo[word].tf = 0;
	}
}

void CalTfAndIdf()
{
	int word;
	for(std::set<long long>::iterator it = wordsInTest.begin(); it != wordsInTest.end(); ++it)
	{
		word = *it;
		wordsInfo[word].idf = log(2 * DATA1_SIZE / wordsInfo[word].idf);
		for(int i = 0; i < DATA1_SIZE; ++i)
			wordsInfo[word].tfidfs[i] *= wordsInfo[word].idf;
	}
}

void ReadTestData()
{
	printf("read test data\n");
	char buf[2500];
	
	FILE* fid;
	errno_t erno = fopen_s(&fid,data353File,"r");
	fgets(buf, 100, fid);

	for(int i = 0; i < DATA0_SIZE; ++i)
	{
		fgets(buf,200,fid);
		strcpy(data353[i].word0 , strtok(buf, ","));
		strcpy(data353[i].word1 , strtok(NULL, ","));

		data353[i].goldSimScore = atof(strtok(NULL, ","));
		data353[i].id = i;

		data353[i].idx0 = trie.GetWordIndex(data353[i].word0);
		data353[i].idx1 = trie.GetWordIndex(data353[i].word1);
	}

	fclose(fid);

	FILE *fi = fopen(ehdataFile, "r");

	char* token, *p;
	int idx = 0, sen[MAX_WORD_IN_SEN], wordPosInSen = 0, senLen;
	while(fgets(buf, 2000, fi) != NULL)
	{
		if(strlen(buf) < 10)
			continue;
		token = strtok_s(buf, "\t", &p);
		for(int i = 0; i < 8; ++i)
		{
			token = strtok_s(NULL,"\t",&p);
			for(int j = 0; token[j] != '\0'; ++j)
				token[j] = tolower(token[j]);
			switch (i)
			{
				case 0: 
					strcpy(ehdata[idx].word0, token);
					ehdata[idx].idx0 = trie.GetWordIndex(ehdata[idx].word0);
					break;

				case 2: 
					strcpy(ehdata[idx].word1, token);
					ehdata[idx].idx1 = trie.GetWordIndex(ehdata[idx].word1);
					break;

				case 4:
					senLen = GetWordsInSen(token, sen);
					if(isTfIdf)
						CalTfScore(senLen, sen, idx * 2);
					GetCtxWords(sen, senLen, ehdata[idx].idx0, ehdata[idx].ctx0, ehdata[idx].ctxSize0);
					break;

				case 5:
					senLen = GetWordsInSen(token, sen);
					if(isTfIdf)
						CalTfScore(senLen, sen, idx * 2 + 1);
					GetCtxWords(sen, senLen, ehdata[idx].idx1, ehdata[idx].ctx1, ehdata[idx].ctxSize1);
					break;

				case 6: 
					ehdata[idx].goldSimScore = atof(token);
					break;

				default:
					break;
			}
		}
		
		ehdata[idx].id = idx;
		idx++;
	}
	
	fclose(fi);

	if(isTfIdf)
		CalTfAndIdf();
}

ele_type ComputeSimScoreWithoutCtx(int word0Idx, int word1Idx)
{
	ele_type score = mysimType == Max? -INT_MAX : 0;
	
	if(word0Idx == -1 || word1Idx == -1)
		return score;

	for(int i = 0; i < wordsInfo[word0Idx].prototypeCnt; ++i)
		for(int j = 0; j < wordsInfo[word1Idx].prototypeCnt; ++j)
		{
			ele_type innerProd = inputEmbedings->dense.col(wordsInfo[word0Idx].idxInInput + i).dot(inputEmbedings->dense.col(wordsInfo[word1Idx].idxInInput + j));
			ele_type cosineSim = innerProd / (inputEmbedings->dense.col(wordsInfo[word0Idx].idxInInput+i).norm() * inputEmbedings->dense.col(wordsInfo[word1Idx].idxInInput + j).norm());

			switch(mysimType)
			{
				case Uniform:
					score += cosineSim;
					break;
				case Weighted:
					score += wordsInfo[word0Idx].prototypePrior[i] * wordsInfo[word1Idx].prototypePrior[j] * cosineSim;
					break;
				case Max:
					score = cosineSim > score? cosineSim: score;
					break;
				default:
					break;
			}
		}

	if(mysimType == Uniform) score /= (wordsInfo[word0Idx].prototypeCnt * wordsInfo[word1Idx].prototypeCnt);
	
	return score;
}

void ReadInputEmbeddings()
{
	printf("read input embeddings\t");
	FileReader reader;
	reader.OpenFile(inputEmbedFile);

	vocabSize = reader.ReadInt();
	inputEmbedSize = reader.ReadInt();
	embedDim = reader.ReadInt();

	inputEmbedings = new EmbedingMatrix(inputEmbedSize,embedDim,false);
	wordsInfo = new wordInfo[vocabSize];

	char word[MAX_WORD_LEN];
	
	int idx = 0;
	for(int i = 0; i < vocabSize; ++i)
	{
		reader.ReadString(word);
		
		trie.Insert(word);
		
		wordsInfo[i].idxInInput = idx;
		strcpy(wordsInfo[i].word, word);
		
		wordsInfo[i].prototypeCnt = reader.ReadInt();

		for(int j = 0; j < wordsInfo[i].prototypeCnt; ++j)
		{
			wordsInfo[i].prototypePrior.push_back(reader.ReadBinaryFloat());

			for(int k = 0; k < embedDim; ++k)
				inputEmbedings->dense(k, idx + j) = reader.ReadBinaryFloat();

			inputEmbedings->dense.col(idx + j).normalize();
		}
		idx += wordsInfo[i].prototypeCnt;
	}
	reader.CloseFile();
}

void ReadOutputEmbeddings()
{
	printf("Read output embeddings\t");

	FileReader reader;
	reader.OpenFile(outEmbedFile);

	int outVocabSize = reader.ReadInt();
	embedDim = reader.ReadInt();

	outputEmbeddings = new EmbedingMatrix(outVocabSize,embedDim,false);

	if(isHs)
	{
		for(int i = 0;i < outVocabSize; ++i)
		{
			for(int j = 0; j < embedDim; ++j)
				outputEmbeddings->dense(j, i) = reader.ReadBinaryFloat();
		}
	}
	else
	{
		char word[MAX_WORD_LEN];
		for(int i = 0; i < outVocabSize; ++i)
		{
			reader.ReadString(word);
			int wordIdx = trie.GetWordIndex(word);
			for(int j = 0;j < embedDim; ++j)
				outputEmbeddings->dense(j, wordIdx) = reader.ReadBinaryFloat();
		}
	}

	reader.CloseFile();
}

void ReadHsStructures()
{
	printf("read hs structures\t");
	FileReader reader;
	reader.OpenFile(hstreeFile);

	vocabSize = reader.ReadInt();

	char word[MAX_WORD_LEN];
	int wordIdx, codeLen;
	for(int i = 0; i < vocabSize; ++i)
	{
		reader.ReadString(word);
		wordIdx = trie.GetWordIndex(word);

		codeLen = reader.ReadInt();

		for(int j = 0; j < codeLen; ++j)
			wordsInfo[wordIdx].hsBinCodes.push_back(reader.ReadInt());

		for(int j = 0; j < codeLen; ++j)
			wordsInfo[wordIdx].hsNodesIdx.push_back(reader.ReadInt());

		wordsInfo[wordIdx].hsCodeLength = codeLen;
	}
}

ele_type ComputeLogProb(int inIdx, int outIdx, int prototypeIdx)
{
	ele_type prob = 0;
	
	if(isHs)
	{
		for(int i = 0; i < wordsInfo[outIdx].hsCodeLength; ++i)
		{
			ele_type innerProduct = inputEmbedings->dense.col(wordsInfo[inIdx].idxInInput + prototypeIdx).dot(outputEmbeddings->dense.col(wordsInfo[outIdx].hsNodesIdx[i]));
			if(wordsInfo[outIdx].hsBinCodes[i] == 0)//If it is 0, then using sigmoid(x) = 1 / (1 + exp(-x))
				innerProduct = -innerProduct;
			prob -= log(1 + exp(innerProduct));
		}
	}
	else
		prob = Util::MyLog(outputEmbeddings->dense.col(outIdx).dot(inputEmbedings->dense.col(wordsInfo[inIdx].idxInInput + prototypeIdx)));

	return prob;
}

void GetPrototypeProbInContext(int wordIdx, int* ctxIds, int ctxSize, ele_type* prob, int doc_id)
{
	if(wordsInfo[wordIdx].prototypeCnt == 1)
	{
		prob[0] = 1;
		return;
	}

	ele_type prob_in[MAX_SENSE_CNT];

	for(int i = 0; i < wordsInfo[wordIdx].prototypeCnt; ++i)
	{
		prob_in[i] = Util::MyLog(wordsInfo[wordIdx].prototypePrior[i]);

		ele_type tfidfSum = 0;
		if(isTfIdf)
			for(int j = 0; j < ctxSize; ++j)
				tfidfSum += wordsInfo[ctxIds[j]].tfidfs[doc_id];

		for(int j = 0; j < ctxSize; ++j)
			prob_in[i] += (isTfIdf? wordsInfo[ctxIds[j]].tfidfs[doc_id] / tfidfSum : 1) * ComputeLogProb(wordIdx, ctxIds[j], i);
	}

	Util::SoftMax(prob_in, prob, wordsInfo[wordIdx].prototypeCnt);

	return;
}

void ComputeSim353Score()
{
	for(int i = 0;i < DATA0_SIZE; ++i)
		data353[i].simScoreWithoutCtx = ComputeSimScoreWithoutCtx(data353[i].idx0, data353[i].idx1);
}

void ComputeEHScores(int currWinSize)
{
	//printf("computing similarity score\n");	
	
	ele_type prob0[MAX_SENSE_CNT], prob1[MAX_SENSE_CNT];

	ele_type aver_eh_score = 0;

	for(int j = 0; j < DATA1_SIZE; ++j)
	{
		if(ehdata[j].idx0 == -1 || ehdata[j].idx1 == -1)
		{
			ehdata[j].simScoreWithCtx = -1;
			badCnt++;

			continue;
		}
		GetPrototypeProbInContext(ehdata[j].idx0, ehdata[j].ctx0, std::min(ehdata[j].ctxSize0, 2 * currWinSize), prob0, j * 2);
		GetPrototypeProbInContext(ehdata[j].idx1, ehdata[j].ctx1, std::min(ehdata[j].ctxSize1, 2 * currWinSize), prob1, j * 2 + 1);
		
		if(mysimType == Max)
		{
			int maxId0 = -1, maxId1 = -1;
			ele_type maxScore = -INT_MAX;

			for(int i = 0; i < wordsInfo[ehdata[j].idx0].prototypeCnt; ++i)
			{
				if(prob0[i] > maxScore)
				{
					maxScore = prob0[i];
					maxId0 = i;
				}
			}

			maxScore = -INT_MAX;
			for(int i = 0; i < wordsInfo[ehdata[j].idx1].prototypeCnt; ++i)
			{
				if(prob1[i] > maxScore)
				{
					maxScore = prob1[i];
					maxId1 = i;
				}
			}

			ehdata[j].simScoreWithCtx
				=inputEmbedings->dense.col(wordsInfo[ehdata[j].idx0].idxInInput + maxId0).dot(inputEmbedings->dense.col(wordsInfo[ehdata[j].idx1].idxInInput + maxId1));

			ehdata[j].simScoreWithCtx /= (inputEmbedings->dense.col(wordsInfo[ehdata[j].idx0].idxInInput + maxId0).norm() * inputEmbedings->dense.col(wordsInfo[ehdata[j].idx1].idxInInput + maxId1).norm());

			aver_eh_score += ehdata[j].simScoreWithCtx;
			continue;
		}

		ele_type score = 0;

		for(int p = 0; p < wordsInfo[ehdata[j].idx0].prototypeCnt; ++p)
			for(int q = 0;q < wordsInfo[ehdata[j].idx1].prototypeCnt; ++q)
			{
				ele_type innerProd = inputEmbedings->dense.col(wordsInfo[ehdata[j].idx0].idxInInput + p).dot(inputEmbedings->dense.col(wordsInfo[ehdata[j].idx1].idxInInput + q));
				ele_type cosineSim = innerProd / (inputEmbedings->dense.col(wordsInfo[ehdata[j].idx0].idxInInput + p).norm() * inputEmbedings->dense.col(wordsInfo[ehdata[j].idx1].idxInInput + q).norm());

				if(mysimType == Uniform)
					score += cosineSim;
				else
					score += prob0[p] * prob1[q] * cosineSim;
			}
			
		ehdata[j].simScoreWithCtx = mysimType == Uniform? score / (wordsInfo[ehdata[j].idx0].prototypeCnt * wordsInfo[ehdata[j].idx1].prototypeCnt) : score;
		aver_eh_score += ehdata[j].simScoreWithCtx;

		//printf("%c%d %.4f", 13, j, ehdata[j].simScoreWithCtx);
	}
	//printf("\n");

	aver_eh_score /= (DATA1_SIZE - badCnt);

	for(int i = 0; i < DATA1_SIZE; ++i)
		if(ehdata[i].simScoreWithCtx == -1)
			ehdata[i].simScoreWithCtx = aver_eh_score;
}

void AnalyzeResults()
{
	std::vector<int> badCases;

	FILE* fi = fopen(badCaseFile, "w");

	int badCaseNum0 = std::min(badCaseNum, DATA0_SIZE);
	ele_type loss = Util::CompareRanks(rankWS, rankGoldWS, DATA0_SIZE, badCases, badCaseNum0);
	fprintf(fi, "Bad cases for ws353 task:\n");
	for(int i = 0; i < badCaseNum0; ++i)
	{
		int idx = badCases[i];
		fprintf(fi, "case: %d\t%s\t%s\tRank at position %d with score %0.4f, you rank it at position %d with score %.4f\n",
			idx, wordsInfo[data353[idx].idx0].word, wordsInfo[data353[idx].idx1].word, rankGoldWS[idx], data353[idx].goldSimScore, rankWS[idx], data353[idx].simScoreWithoutCtx);
	}
	fprintf(fi, "Total loss these bad cases bring is %.4f\n", loss);

	int badCaseNum1 = std::min(badCaseNum, DATA1_SIZE);
	loss = Util::CompareRanks(bestRankEh, rankGoldEH, DATA1_SIZE, badCases, badCaseNum1);

	fprintf(fi, "\n\nBad cases for eh task:\n");
	for(int i = 0; i < badCaseNum1; ++i)
	{
		int idx = badCases[i];
		fprintf(fi, "\ncase: %d\t%s\t%s\tRank at position %d with score %0.4f, you rank it at position %d with score %.4f\n",
			idx + 1, wordsInfo[ehdata[idx].idx0].word, wordsInfo[ehdata[idx].idx1].word, rankGoldEH[idx], ehdata[idx].goldSimScore, rankEH[idx], ehdata[idx].simScoreWithCtx);

		for(int j = 0; j < ehdata[idx].ctxSize0; ++j)
			fprintf(fi, "%s ", wordsInfo[ehdata[idx].ctx0[j]].word);
		fprintf(fi, "\n");

		for(int j = 0; j < ehdata[idx].ctxSize1; ++j)
			fprintf(fi, "%s ", wordsInfo[ehdata[idx].ctx1[j]].word);
		fprintf(fi, "\n");
	}

	fprintf(fi, "Total loss these bad cases bring is %.4f\n", loss);
	fclose(fi);
}

int main(int argc, char* argv[])
{
	if(!ParseCmdArgs(argc, argv))
		return -1;

	ReadInputEmbeddings();
	ReadOutputEmbeddings();
	if(isHs)
		ReadHsStructures();
	if(isStopWords)
		ReadStopWords();
	
	ReadTestData();
	
	ComputeSim353Score();
	std::qsort(data353, DATA0_SIZE, sizeof(wordPair), CmpWithOutCtx);

	for(int i = 0; i < DATA0_SIZE; ++i)
		rankWS[data353[i].id] = i;

	std::qsort(data353, DATA0_SIZE, sizeof(wordPair), CmpWithGold);
	for(int i = 0;i < DATA0_SIZE; ++i)
		rankGoldWS[data353[i].id] = i;
	ele_type pcorrEM0 = Util::ComputeSRankCorr(rankGoldWS, rankWS, DATA0_SIZE);


	std::qsort(ehdata, DATA1_SIZE, sizeof(wordPair), CmpWithGold);

	for(int i = 0; i < DATA1_SIZE; ++i)
		rankGoldEH[ehdata[i].id] = i;

	ele_type pcorrEM1 = -1000;

	int best_wins_size;
	for(int cwindow = 1; cwindow <= (mysimType == simType::Uniform? 1: MAX_WINDOW_SIZE); ++ cwindow)
	{
		ComputeEHScores(cwindow);	

		std::qsort(ehdata, DATA1_SIZE, sizeof(wordPair), CmpWithCtx);
		for(int i = 0; i < DATA1_SIZE; ++i)
			rankEH[ehdata[i].id] = i;
	
		ele_type pcorr_tmp = Util::ComputeSRankCorr(rankGoldEH, rankEH, DATA1_SIZE);

		if(pcorr_tmp > pcorrEM1)
		{
			pcorrEM1 = pcorr_tmp;
			best_wins_size = cwindow;
			if(badCaseNum > 0)
			{
				memcpy(bestRankEh, rankEH, DATA1_SIZE * sizeof(*bestRankEh));
				for(int i = 0; i < DATA1_SIZE; ++i)
					bestEhSimScores[ehdata[i].id] = ehdata[i].simScoreWithCtx; 
			}
		}
		printf("%d %.4f\t", cwindow, pcorr_tmp);
	}

	printf("\nws353 score:%.4f eh score:%.4f best window: %d bad count:%d\n\n", pcorrEM0, pcorrEM1, best_wins_size, badCnt);
	
	if(badCaseNum > 0)
	{
		printf("Extracting bad cases\n");
		std::qsort(data353, DATA0_SIZE, sizeof(wordPair), CmpWithIdx);
		std::qsort(ehdata, DATA1_SIZE, sizeof(wordPair), CmpWithIdx);
		for(int i = 0; i < DATA1_SIZE; ++i)
			ehdata[i].simScoreWithCtx = bestEhSimScores[ehdata[i].id];

		AnalyzeResults();
	}

	delete [] wordsInfo;
	
	return 1;
}
