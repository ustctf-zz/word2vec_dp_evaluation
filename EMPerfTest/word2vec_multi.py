import logging
import sys
import os
import numpy as np
from numpy.linalg import norm
from operator import itemgetter
REAL = np.float32

def cosSim(a, b):
    return np.dot(a, b) / norm(a) / norm(b)


class Vocab_item:
    def __init__(self, index, sense_count):
        self.index = index;
        self.sense_count = sense_count


class Word2Vec_Multiprototype:
    def __init__(self):
        """
        Initialize the model from an iterable of `sentences`. Each sentence is a
        list of words (utf8 strings) that will be used for training.

        The `sentences` iterable can be simply a list, but for larger corpora,
        consider an iterable that streams the sentences directly from disk/network.
        See :class:`BrownCorpus`, :class:`Text8Corpus` or :class:`LineSentence` in
        this module for such examples.

        If you don't supply `sentences`, the model is left uninitialized -- use if
        you plan to initialize it in some other way.

        `size` is the dimensionality of the feature vectors.
        `window` is the maximum distance between the current and predicted word within a sentence.
        `alpha` is the initial learning rate (will linearly drop to zero as training progresses).
        `seed` = for the random number generator.
        `min_count` = ignore all words with total frequency lower than this.
        `workers` = use this many worker threads to train the model (=faster training with multicore machines)

        """
        self.vocab = {}  # mapping from a word (string) to a Vocab object
        self.index2word = []  # map from a word's matrix index (int) to word (string)
        self.embedding_index = []
        self.size = 0
        self.vocab_size = 0
        self.embedding_count = 0
        
    @classmethod
    def loadWordEmbedding(cls, fname, binary=True):
        """
        Load the input-hidden weight matrix from the original C word2vec-tool format.

        Note that the information loaded is incomplete (the binary tree is missing),
        so while you can query for word similarity etc., you cannot continue training
        with a model loaded this way.

        """
        print "loading projection weights from %s" % (fname)
        with open(fname, 'rb') as fin:
            header = fin.readline()
            vocab_size, embedding_count, layer1_size  = map(int, header.split())
            result = Word2Vec_Multiprototype()
            result.size = layer1_size
            result.embedding_count = embedding_count
            result.syn0 = np.zeros((embedding_count, layer1_size), dtype=REAL)
            result.sense_prior = np.zeros(embedding_count, dtype=REAL)
            if binary:
                binary_len = np.dtype(REAL).itemsize * layer1_size
                embedding_idx = 0
                # Loop through each word
                for line_no in xrange(vocab_size):
                    result.embedding_index.append(embedding_idx)
                    # mixed text and binary: read text first, then binary
                    word = []
                    # Read word 
                    while True:
                        ch = fin.read(1)
                        if ch == ' ':
                            word = ''.join(word)
                            break
                        if ch != '\n':  # ignore newlines in front of words (some binary files have newline, some not)
                            word.append(ch)
                    
                    
                    # Read sense_count
                    sense_count = []
                    while True:
                        ch = fin.read(1)
                        if ch == ' ':
                            sense_count = ''.join(sense_count)
                            break
                        if ch != '\n':  # ignore newlines in front of words (some binary files have newline, some not)
                            sense_count.append(ch)
                    sense_count = int(sense_count)
                    if word not in result.vocab:
                        result.vocab[word] = Vocab_item(line_no, sense_count)
                    #result.vocab[word] = Vocab(index=line_no, count=vocab_size - line_no)
                    for i in xrange(sense_count):
                        result.sense_prior[embedding_idx] = np.fromstring(fin.read(np.dtype(REAL).itemsize), dtype=REAL)
                        result.syn0[embedding_idx] = np.fromstring(fin.read(binary_len), dtype=REAL)
                        embedding_idx += 1
                        result.index2word.append([word, i])
                        
            else:
                for line_no, line in enumerate(fin):
                    parts = line.split()
                    assert len(parts) == layer1_size + 1
                    word, weights = parts[0], map(REAL, parts[1:])
                    result.vocab[word] = Vocab(index=line_no, count=vocab_size - line_no)
                    result.index2word.append(word)
                    result.syn0[line_no] = weights
        print "loaded %s matrix from %s" % (result.syn0.shape, fname)
        return result

    def loadOutputEmbedding(self, fname, binary=True):
        print "loading projection weights from %s" % (fname)
        with open(fname, 'rb') as fin:
            header = fin.readline()
            vocab_size, layer1_size, model = map(int, header.split())
            self.syn1 = np.zeros((vocab_size, layer1_size), dtype=REAL)
            if binary and model == 0:
                binary_len = np.dtype(REAL).itemsize * layer1_size
                embedding_idx = 0
                # Read embedding one by one
                for embedding_no in xrange(vocab_size):
                    #result.vocab[word] = Vocab(index=line_no, count=vocab_size - line_no)
                    result.syn0[embedding_idx] = np.fromstring(fin.read(binary_len), dtype=REAL)
                        
            else:
                pass
        print "loaded %s matrix from %s" % (result.syn0.shape, fname)
        return result
    
    def getEmbedding(self, word, sense_idx = -1):
        if word in self.vocab:
            result = []
            sense_count = self.vocab[word].sense_count
            vocab_item = self.vocab[word]
            start_idx = self.embedding_index[vocab_item.index]
            if sense_idx < 0:
                for i in xrange(sense_count):
                    item = [self.sense_prior[start_idx + i], self.syn0[start_idx + i]]
                    result.append(item)
            else:
                result = [[self.sense_prior[start_idx + sense_idx], self.syn0[start_idx + sense_idx]]]
            return result
        else:
            return None

    def getPrior(self, word):
        if word in self.vocab:
            sense_count = self.vocab[word].sense_count
            priors = []
            vocab_item = self.vocab[word]
            start_idx = self.embedding_index[vocab_item.index]
            for i in xrange(sense_count):
                priors.append(self.sense_prior[start_idx + i])
            return [sense_count, priors]
        else:
            return None
    def avgSim(self, wordL, wordR):
        wordL_embedding = self.getEmbedding(wordL)
        wordR_embedding = self.getEmbedding(wordR)

        if not wordL_embedding:
            print 'Embedding of first word not found'
            return -1;
        elif not wordR_embedding:
            print 'Embedding of second word not found'
            return -1;
        else:
            similarity = 0.0
            for embeddingL in wordL_embedding:
                for embeddingR in wordR_embedding:
                    similarity += cosSim(embeddingL[1], embeddingR[1])
            return similarity / len(wordL_embedding) / len(wordR_embedding)

    def maxSim(self, wordL, wordR):
        wordL_embedding = self.getEmbedding(wordL)
        wordR_embedding = self.getEmbedding(wordR)

        if not wordL_embedding:
            print 'Embedding of first word not found'
            return -1;
        elif not wordR_embedding:
            print 'Embedding of second word not found'
            return -1;
        else:
            similarity = 0.0
            max_sim = -1.0
            for embeddingL in wordL_embedding:
                for embeddingR in wordR_embedding:
                    similarity = cosSim(embeddingL[1], embeddingR[1])
                    if similarity > max_sim:
                        max_sim = similarity
            return max_sim

    def avgSim_P(self, wordL, wordR):
        wordL_embedding = self.getEmbedding(wordL)
        wordR_embedding = self.getEmbedding(wordR)
        wordL_prior_idx = self.embedding_index[self.vocab[wordL].index]
        wordR_prior_idx = self.embedding_index[self.vocab[wordR].index]

        if not wordL_embedding:
            print 'Embedding of first word not found'
            return -1
        elif not wordR_embedding:
            print 'Embedding of second word not found'
            return -1
        else:
            similarity = 0.0
            for L_idx in xrange(len(wordL_embedding)):
                for R_idx in xrange(len(wordR_embedding)):
                    similarity = cosSim(wordL_embedding[L_idx][1], wordR_embedding[R_idx][1]) * wordL_embedding[L_idx][0] * wordR_embedding[R_idx][0]
            return similarity

    def maxSim_P(self, wordL, wordR):
        wordL_embedding = self.getEmbedding(wordL)
        wordR_embedding = self.getEmbedding(wordR)

        if not wordL_embedding:
            print 'Embedding of first word not found'
            return -1
        elif not wordR_embedding:
            print 'Embedding of second word not found'
            return -1
        else:
            similarity = 0.0
            max_sim = -1.0
            max_L_sense = 0
            max_L_prior = 0.0
            max_R_sense = 0
            max_R_prior = 0.0
            for L_idx in xrange(len(wordL_embedding)):
                if wordL_embedding[L_idx][0] > max_L_prior:
                    max_L_prior = wordL_embedding[L_idx][0]
                    max_L_sense = L_idx

            for R_idx in xrange(len(wordR_embedding)):
                if wordR_embedding[R_idx][0] > max_R_prior:
                    max_R_prior = wordR_embedding[R_idx][0]
                    max_R_sense = R_idx
                    
            return cosSim(wordL_embedding[max_L_sense][1], wordR_embedding[max_R_sense][1])

    def mostSimilarEmbedding(self, word, sense = 0, topn=10):
        print 'Finding Top %d most similar embedding of sense %d of word \'%s\'' % (topn, sense, word)
        if word not in self.vocab:
            print 'Error: word %s not in vaocabulary' % word
            return None

        if sense > self.vocab[word].sense_count:
            print 'Error: Sense %d of word %s does not exist00' % (sense, word)
            return None

        score_list = []
        query_embedding = self.getEmbedding(word, sense)
        # Calculate cosine similarity
        for i in xrange(len(self.syn0)):
            if self.index2word[i][0] != word:
                score = cosSim(query_embedding[0][1], self.syn0[i])
                score_list.append(score)
            else:
                score_list.append(-99999)
        # Sort result
        sorted_score = sorted(enumerate(score_list), key=itemgetter(1), reverse = True)
        print 'Top %d most similar embedding of sense %d of word \'%s\'' % (topn, sense, word)
        for i in xrange(topn):
            top_word = self.index2word[sorted_score[i][0]]
            print '\'%s\'\tsense %d\tscore: %f' %(top_word[0], top_word[1], sorted_score[i][1])
            
    def mostSimilarWord(self, word, sense = 0, topn=10):
        if word not in self.vocab:
            print 'Error: word %s not in vaocabulary' % word
            return None

        if sense > self.vocab[word].sense_count:
            print 'Error: Sense %d of word %s does not exist00' % (sense, word)
            return None

        print 'Finding Top %d most similar word of sense %d of word \'%s\'' % (topn, sense, word)
        score_list = []
        query_embedding = self.getEmbedding(word, sense)
        # Calculate cosine similarity
        for i in xrange(len(self.syn0)):
            if self.index2word[i][0] != word:
                score = cosSim(query_embedding[0][1], self.syn0[i])
                score_list.append(score)
            else:
                score_list.append(-99999)
        # Sort result
        sorted_score = sorted(enumerate(score_list), key=itemgetter(1), reverse = True)
        print 'Top %d most similar word of sense %d of word \'%s\'' % (topn, sense, word)
        similar_words = []
        for score in sorted_score:
            top_word = self.index2word[score[0]]
            if top_word[0] not in similar_words:
                similar_words.append(top_word[0])
                print '\'%s\'\tscore: %f' %(top_word[0], score[1])
            if len(similar_words) >= topn:
                break
                

    def similarity(self, wordL, wordR, senseL = [], senseR = []):
        L_embedding = self.getEmbedding(wordL)
        R_embedding = self.getEmbedding(wordR)

        if type(senseL) == int:
            rangeL = [senseL]
        elif type(senseL) == list:
            if not senseL:
                rangeL = range(len(L_embedding))
            else:
                rangeL = senseL

        if type(senseR) == int:
            rangeR = [senseR]
        elif type(senseR) == list:
            if not senseR:
                rangeR = range(len(R_embedding))
            else:
                rangeR = senseR
            
        for L_idx in rangeL:
            for R_idx in rangeR:
                similarity = cosSim(L_embedding[L_idx][1], R_embedding[R_idx][1])
                print 'sense %d of word \'%s\' and sense %d of word \'%s\'' % (L_idx, wordL, R_idx, wordR)
                print similarity
                
            
            
