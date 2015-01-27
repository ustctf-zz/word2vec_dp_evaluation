
..\x64\Release\EMPerfTest.exe combined.csv ratings.txt lr_0.025_wikiEH_bs1_epoch3_window5.bin lr_0.025_wikiEH_bs1_epoch3_window5_outlayer.bin lr_0.025_wikiEH_bs1_epoch3_window5_huff.txt -s 2 -sw 1 stop_words_eh.txt -tfidf 0 -d 50 debug.txt

rem "..\x64\Release\EMPerfTest - Copy.exe" combined.csv ratings.txt wikiEH_bs1_epoch3_window5.bin wikiEH_bs1_epoch3_window5_outlayer.bin wikiEH_bs1_epoch3_window5_huff.txt -c 10 -s 0

..\x64\Release\EMPerfTest.exe combined.csv ratings.txt lr_0.025_wikiEH_bs1_epoch3_window5.bin lr_0.025_wikiEH_bs1_epoch3_window5_outlayer.bin lr_0.025_wikiEH_bs1_epoch3_window5_huff.txt -s 2 -sw 1 stop_words_eh.txt -tfidf 0

rem "..\x64\Release\EMPerfTest - Copy.exe" combined.csv ratings.txt wiki2010_bs1_epoch5_window5.bin wiki2010_bs1_epoch5_window5_outlayer.bin wiki2010_bs1_epoch5_window5_huff.txt -c 3 -s 1 -sw 1 stop_words_eh.txt