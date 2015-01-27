@echo off

set test=text8
set sw_file=stop_words_eh.txt
rem set hs_file=text8_bs1_epoch1_huff.txt
rem set in_emb_file=text8_bs1_epoch1.bin
rem set out_emb_file=text8_bs1_epoch1_outlayer.bin

set stopwords=1

set hs_file=text8_bs1_epoch1_window5_huff.txt
set in_emb_file=text8_bs1_epoch1_window5.bin
set out_emb_file=text8_bs1_epoch1_window5_outlayer.bin
.\x64\Release\LMPerfTest.exe -emb %in_emb_file% -test %test% -hsfile %hs_file% -stopwords %stopwords% -sw_file %sw_file% -out_emb %out_emb_file%
