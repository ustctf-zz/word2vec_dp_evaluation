
#include "utility.h"
#include <Windows.h>
#include <iostream>
#include <cstdio>


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


	FILE* fid;
	fopen_s(&fid,filename, "rb");
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