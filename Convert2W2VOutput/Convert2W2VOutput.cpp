// Convert2W2VOutput.cpp : Defines the entry point for the console application.
//

#include "../EMPerfTest/utility.h"

//Convert the bin file outputed by EM algorithm to the original output file of W2V

//argv[1]: the bin file of EM algorithm
int main(int argc, char* argv[])
{
	char in_file_name[MAX_FILENAME_LEN];
	char out_file_name[MAX_FILENAME_LEN];
	strcpy(in_file_name, argv[1]);

	sprintf(out_file_name, "%s_w2v_out.bin", in_file_name);
	FILE* fout = fopen(out_file_name, "wb");

	FileReader reader;
	reader.OpenFile(in_file_name);

	int vocabSize = reader.ReadInt();
	int inputEmbedSize = reader.ReadInt();
	int embedDim = reader.ReadInt();
	fprintf(fout, "%d %d\n", vocabSize, embedDim);

	char word[MAX_WORD_LEN];

	int idx = 0;
	ele_type* p_emb = (ele_type*)calloc(embedDim, sizeof(ele_type));
	ele_type* curr_emb = (ele_type*)calloc(embedDim, sizeof(ele_type));
	ele_type prior;

	for (int i = 0; i < vocabSize; ++i)
	{
		reader.ReadString(word);

		fprintf(fout, "%s ", word);

		int prototypeCnt = reader.ReadInt();

		memset(p_emb, 0, sizeof(ele_type)* embedDim);

		for (int j = 0; j < prototypeCnt; ++j)
		{
			prior = reader.ReadBinaryFloat();

			for (int k = 0; k < embedDim; ++k)
			{
				curr_emb[k] = reader.ReadBinaryFloat();
				p_emb[k] += (prior) * curr_emb[k];
				//p_emb[k] += (1.0 / prototypeCnt)* curr_emb[k];
			}
		}

		for (int k = 0; k < embedDim; ++k)
			fwrite(p_emb + k, sizeof(ele_type), 1, fout);
		fprintf(fout, "\n");
	}

	reader.CloseFile();
	fclose(fout);
	return 0;
}

