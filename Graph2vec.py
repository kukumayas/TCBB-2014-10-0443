# -*- coding: utf-8 -*-

from gensim.utils import SaveLoad


"""
Deep learning via word2vec's "skip-gram and CBOW models", using either
hierarchical softmax or negative sampling [1]_ [2]_.

The training algorithms were originally ported from the C package https://code.google.com/p/word2vec/
and extended with additional functionality.

For a blog tutorial on gensim word2vec, with an interactive web app trained on GoogleNews, visit http://radimrehurek.com/2014/02/word2vec-tutorial/

**Make sure you have a C compiler before installing gensim, to use optimized (compiled) word2vec training**
(70x speedup compared to plain NumPy implemenation [3]_).

Initialize a model with e.g.::

>>> model = Word2Vec(sentences, size=100, window=5, min_count=5, workers=4)

Persist a model to disk with::

>>> model.save(fname)
>>> model = Word2Vec.load(fname)  # you can continue training with the loaded model!

The model can also be instantiated from an existing file on disk in the word2vec C format::

  >>> model = Word2Vec.load_word2vec_format('/tmp/vectors.txt', binary=False)  # C text format
  >>> model = Word2Vec.load_word2vec_format('/tmp/vectors.bin', binary=True)  # C binary format

You can perform various syntactic/semantic NLP word tasks with the model. Some of them
are already built-in::

  >>> model.most_similar(positive=['woman', 'king'], negative=['man'])
  [('queen', 0.50882536), ...]

  >>> model.doesnt_match("breakfast cereal dinner lunch".split())
  'cereal'

  >>> model.similarity('woman', 'man')
  0.73723527

  >>> model['computer']  # raw numpy vector of a word
  array([-0.00449447, -0.00310097,  0.02421786, ...], dtype=float32)

and so on.

If you're finished training a model (=no more updates, only querying), you can do

  >>> model.init_sims(replace=True)

to trim unneeded model memory = use (much) less RAM.

Note that there is a :mod:`gensim.models.phrases` module which lets you automatically
detect phrases longer than one word. Using phrases, you can learn a word2vec model
where "words" are actually multiword expressions, such as `new_york_times` or `financial_crisis`:

>>> bigram_transformer = gensim.models.Phrases(sentences)
>>> model = Word2Vec(bigram_transformed[sentences], size=100, ...)

.. [1] Tomas Mikolov, Kai Chen, Greg Corrado, and Jeffrey Dean. Efficient Estimation of Word Representations in Vector Space. In Proceedings of Workshop at ICLR, 2013.
.. [2] Tomas Mikolov, Ilya Sutskever, Kai Chen, Greg Corrado, and Jeffrey Dean. Distributed Representations of Words and Phrases and their Compositionality.
       In Proceedings of NIPS, 2013.
.. [3] Optimizing word2vec in gensim, http://radimrehurek.com/2013/09/word2vec-in-python-part-two-optimizing/
"""

import logging
import sys
import os
import heapq
import time
from copy import deepcopy
import threading
try:
    from queue import Queue
except ImportError:
    from Queue import Queue

from numpy import exp, dot, zeros, outer, random, dtype, float32 as REAL, \
    uint32, seterr, array, uint8, vstack, argsort, fromstring, sqrt, newaxis, \
    ndarray, empty, prod, sum, transpose, matrix

logger = logging.getLogger("gensim.models.word2vec")


from gensim import utils, matutils  # utility fnc for pickling, common scipy operations etc
from six import iteritems, itervalues, string_types
from six.moves import xrange


# failed... fall back to plain numpy (20-80x slower training than the above)
FAST_VERSION = -1

def train_relation_sg(model, relation, alpha, work=None):
    labels = []
    train_sg_pair(model, relation[0], relation[1], relation[2], relation[3], alpha, labels)
    return 2

def train_sg_pair(model, target, rel_t, rel_d, word, alpha, labels):
    l1 = model.syn0[word.index]
    neu1e = zeros(l1.shape)
    

    l2a = deepcopy(model.syn1[target.point])  # 2d matrix, codelen x layer1_size
    fa = 1.0 / (1.0 + exp(-dot(l1, l2a.T)))  # propagate hidden -> output
    ga = (1 - target.code - fa) * alpha  # vector of error gradients multiplied by the learning rate
    model.syn1[target.point] += outer(ga, l1)  # learn hidden -> output
    update_rel_target = dot(ga, l2a)
    
    
    ewx_rel_t = exp(dot(model.syn1rel_type, model.syn0[word.index]))
    P_t = ewx_rel_t / sum(ewx_rel_t)
    update_rel_t = alpha * dot(model.syn1rel_type.T, rel_t - P_t) 
    model.syn1rel_type += alpha * outer(rel_t - P_t, model.syn0[word.index])
    
    ewx_rel_d = exp(dot(model.syn1rel_direction, model.syn0[word.index]))
    P_d = ewx_rel_d / sum(ewx_rel_d)
    update_rel_d = alpha * dot(model.syn1rel_direction.T, rel_d - P_d) 
    model.syn1rel_direction += alpha * outer(rel_d - P_d, model.syn0[word.index])
    
    
    neu1e += update_rel_target  # save error
    neu1e += update_rel_t
    neu1e += update_rel_d
    model.syn0[word.index] += neu1e  # learn input -> hidden
    

    return neu1e




class Vocab(object):

    def __init__(self, **kwargs):
        self.count = 0
        self.__dict__.update(kwargs)

    def __lt__(self, other):  # used for sorting in a priority queue
        return self.count < other.count

    def __str__(self):
        vals = ['%s:%r' % (key, self.__dict__[key]) for key in sorted(self.__dict__) if not key.startswith('_')]
        return "<" + ', '.join(vals) + ">"

class RepeatCorpusNTimes(SaveLoad):

    def __init__(self, relation_file, n, vocab, relation_type, relation_direction):

        self.relation_file = relation_file
        self.n = n
        self.vocab = vocab
        self.relation_type = relation_type
        self.relation_direction = relation_direction
        self.len_rel_t = len(self.relation_type)
        self.len_rel_d = len(self.relation_direction)
    
    def rel_type_vector(self, rel_type):
        a = zeros(self.len_rel_t, dtype=uint8)
        a[self.relation_type[rel_type]] = 1
        return a
    
    def rel_direction_vector(self, rel_direction):
        a = zeros(self.len_rel_d, dtype=uint8)
        a[self.relation_direction[rel_direction]] = 1
        return a
        
    
    
    def __iter__(self):
        for _ in xrange(self.n):
            for relation in open(self.relation_file):
                tups = relation.strip().split(' ')
                yield (self.vocab[tups[0]], self.rel_type_vector(tups[1]), self.rel_direction_vector(tups[2]), self.vocab[tups[3]])
                
                
                
class Graph2Vec(utils.SaveLoad):        
    
    def __init__(self, relation_file=None, size=100, alpha=0.025, seed=1, workers=1, min_alpha=0.0001, hashfxn=hash, iter=1):
        """
        Initialize the model from an iterable of `sentences`. Each sentence is a
        list of words (unicode strings) that will be used for training.

        The `sentences` iterable can be simply a list, but for larger corpora,
        consider an iterable that streams the sentences directly from disk/network.
        See :class:`BrownCorpus`, :class:`Text8Corpus` or :class:`LineSentence` in
        this module for such examples.

        If you don't supply `sentences`, the model is left uninitialized -- use if
        you plan to initialize it in some other way.

       `size` is the dimensionality of the feature vectors.

        `window` is the maximum distance between the current and predicted word within a sentence.

        `alpha` is the initial learning rate (will linearly drop to zero as training progresses).

        `seed` = for the random number generator. Initial vectors for each
        word are seeded with a hash of the concatenation of word + str(seed).

        `workers` = use this many worker threads to train the model (=faster training with multicore machines).

        `hashfxn` = hash function to use to randomly initialize weights, for increased
        training reproducibility. Default is Python's rudimentary built in hash function.

        `iter` = number of iterations (epochs) over the corpus.

        """
        self.vocab = {}  # mapping from a word (string) to a Vocab object
        self.relation_type = {}
        self.relation_direction = {}
        self.index2word = []  # map from a word's matrix index (int) to word (string)
        self.layer1_size = int(size)
        if size % 4 != 0:
            logger.warning("consider setting layer size to a multiple of 4 for greater performance")
        self.alpha = float(alpha)
        self.seed = seed
        self.workers = workers
        self.min_alpha = min_alpha
        self.hashfxn = hashfxn
        self.iter = iter
        if relation_file is not None:
            self.build_vocab(relation_file)
            relations = RepeatCorpusNTimes(relation_file, iter, self.vocab, self.relation_type, self.relation_direction)
            self.train(relations)

    def create_binary_tree(self):
        """
        Create a binary Huffman tree using stored vocabulary word counts. Frequent words
        will have shorter binary codes. Called internally from `build_vocab()`.

        """
        logger.info("constructing a huffman tree from %i words" % len(self.vocab))

        # build the huffman tree
        heap = list(itervalues(self.vocab))
        heapq.heapify(heap)
        for i in xrange(len(self.vocab) - 1):
            min1, min2 = heapq.heappop(heap), heapq.heappop(heap)
            heapq.heappush(heap, Vocab(count=min1.count + min2.count, index=i + len(self.vocab), left=min1, right=min2))

        # recurse over the tree, assigning a binary code to each vocabulary word
        if heap:
            max_depth, stack = 0, [(heap[0], [], [])]
            while stack:
                node, codes, points = stack.pop()
                if node.index < len(self.vocab):
                    # leaf node => store its path from the root
                    node.code, node.point = codes, points
                    max_depth = max(len(codes), max_depth)
                else:
                    # inner node => continue recursion
                    points = array(list(points) + [node.index - len(self.vocab)], dtype=uint32)
                    stack.append((node.left, array(list(codes) + [0], dtype=uint8), points))
                    stack.append((node.right, array(list(codes) + [1], dtype=uint8), points))

            logger.info("built huffman tree with maximum node depth %i" % max_depth)

    def build_vocab(self, relation_file):
        """
        Build vocabulary from a sequence of sentences (can be a once-only generator stream).
        Each sentence must be a list of unicode strings.

        """
        logger.info("collecting all words and their counts")
        vocab, relation_type, relation_direction = self._vocab_from(relation_file)
        # assign a unique index to each word
        self.vocab, self.index2word = {}, []
        for word, v in iteritems(vocab):
            v.index = len(self.vocab)
            self.index2word.append(word)
            self.vocab[word] = v
            
        self.relation_type = relation_type
        self.index2relation_type = dict((v, k) for k, v in self.relation_type.iteritems())
        
        self.relation_direction = relation_direction
        self.index2relation_direction = dict((v, k) for k, v in self.relation_direction.iteritems())
        
        logger.info("total %i word types " % len(self.vocab))
        
        self.create_binary_tree()
        # precalculate downsampling thresholds
        self.reset_weights()

    @staticmethod
    def _vocab_from(relation_file):
        relation_no, vocab, relation_type, relation_direction = -1, {}, {}, {}
        total_words = 0
    
        for relation in open(relation_file, 'rb'):
            relation_no += 1
            target, type, direction, word = relation.strip().split(' ')
            
            total_words += 2
            if target in vocab:
                vocab[target].count += 1
            else:
                vocab[target] = Vocab(count=1)
                
            if word in vocab:
                vocab[word].count += 1
            else:
                vocab[word] = Vocab(count=1)
            
            if type not in relation_type:
                relation_type[type] = len(relation_type)
            
            if direction not in relation_direction:
                relation_direction[direction] = len(relation_direction)
            
            
            
        logger.info("collected %i word types from a corpus of %i words, %i relation types, %i relation direction types,  and %i relations" % 
                    (len(vocab), total_words, len(relation_type), len(relation_direction), relation_no + 1))
        return vocab, relation_type, relation_direction


    def _get_job_words(self, alpha, work, job, neu1):
        return sum(train_relation_sg(self, relation, alpha, work) for relation in job)
    
    
    def _prepare_relations(self, relations):
        for relation in relations:
            # avoid calling random_sample() where prob >= 1, to speed things up a little:
            sampled = [self.vocab[relation[0]], ]
            yield sampled

    def train(self, relations, total_words=None, word_count=0, chunksize=100):
        """
        relations  现在是4元组了，target tag direction word
        """
        """
        Update the model's neural weights from a sequence of relations (can be a once-only generator stream).
        Each relation must be a tuple.

        """
        if FAST_VERSION < 0:
            import warnings
            warnings.warn("C extension compilation failed, training will be slow. Install a C compiler and reinstall gensim for fast training.")
        logger.info("training model with %i workers on %i vocabulary and %i features, " % 
            (self.workers, len(self.vocab), self.layer1_size))

        if not self.vocab:
            raise RuntimeError("you must first build vocabulary before training the model")

        start, next_report = time.time(), [1.0]
        word_count = [word_count]
        total_words = total_words or int(sum(v.count for v in itervalues(self.vocab)) * self.iter)
        jobs = Queue(maxsize=2 * self.workers)  # buffer ahead only a limited number of jobs.. this is the reason we can't simply use ThreadPool :(
        lock = threading.Lock()  # for shared state (=number of words trained so far, log reports...)

        def worker_train():
            """Train the model, lifting lists of relations from the jobs queue."""
            work = zeros(self.layer1_size, dtype=REAL)  # each thread must have its own work memory
            neu1 = matutils.zeros_aligned(self.layer1_size, dtype=REAL)
            
            while True:
                job = jobs.get()
                if job is None:  # data finished, exit
                    break
                # update the learning rate before every job
                alpha = max(self.min_alpha, self.alpha * (1 - 1.0 * word_count[0] / total_words))
                # how many words did we train on? out-of-vocabulary (unknown) words do not count
                job_words = self._get_job_words(alpha, work, job, neu1)
                with lock:
                    word_count[0] += job_words
                    elapsed = time.time() - start
                    if elapsed >= next_report[0]:
                        logger.info("PROGRESS: at %.2f%% words, alpha %.05f, %.0f words/s" % 
                            (100.0 * word_count[0] / total_words, alpha, word_count[0] / elapsed if elapsed else 0.0))
                        next_report[0] = elapsed + 1.0  # don't flood the log, wait at least a second between progress reports

        workers = [threading.Thread(target=worker_train) for _ in xrange(self.workers)]
        for thread in workers:
            thread.daemon = True  # make interrupting the process with ctrl+c easier
            thread.start()

        # convert input strings to Vocab objects (eliding OOV/downsampled words), and start filling the jobs queue
        for job_no, job in enumerate(utils.grouper(relations, chunksize)):
            logger.debug("putting job #%i in the queue, qsize=%i" % (job_no, jobs.qsize()))
            jobs.put(job)
        logger.info("reached the end of input; waiting to finish %i outstanding jobs" % jobs.qsize())
        for _ in xrange(self.workers):
            jobs.put(None)  # give the workers heads up that they can finish -- no more work!

        for thread in workers:
            thread.join()

        elapsed = time.time() - start
        logger.info("training on %i words took %.1fs, %.0f words/s" % 
            (word_count[0], elapsed, word_count[0] / elapsed if elapsed else 0.0))
        self.syn0norm = None
        return word_count[0]

    def reset_weights(self): 
        """Reset all projection weights to an initial (untrained) state, but keep the existing vocabulary."""
        logger.info("resetting layer weights")
        self.syn0 = empty((len(self.vocab), self.layer1_size), dtype=REAL)
        # randomize weights vector by vector, rather than materializing a huge random matrix in RAM at once
        for i in xrange(len(self.vocab)):
            # construct deterministic seed from word AND seed argument
            # Note: Python's built in hash function can vary across versions of Python
            random.seed(uint32(self.hashfxn(self.index2word[i] + str(self.seed))))
            self.syn0[i] = (random.rand(self.layer1_size) - 0.5) / self.layer1_size
   
        self.syn1 = zeros((len(self.vocab), self.layer1_size), dtype=REAL)
        
        self.syn1rel_type = zeros((len(self.relation_type), self.layer1_size), dtype=REAL)
        
        self.syn1rel_direction = zeros((len(self.relation_direction), self.layer1_size), dtype=REAL)
        
        self.syn0norm = None


    def save_word2vec_format(self, fname, fvocab=None, binary=False):
        """
        Store the input-hidden weight matrix in the same format used by the original
        C word2vec-tool, for compatibility.

        """
        if fvocab is not None:
            logger.info("Storing vocabulary in %s" % (fvocab))
            with utils.smart_open(fvocab, 'wb') as vout:
                for word, vocab in sorted(iteritems(self.vocab), key=lambda item:-item[1].count):
                    vout.write(utils.to_utf8("%s %s\n" % (word, vocab.count)))
        logger.info("storing %sx%s projection weights into %s" % (len(self.vocab), self.layer1_size, fname))
        assert (len(self.vocab), self.layer1_size) == self.syn0.shape
        with utils.smart_open(fname, 'wb') as fout:
            fout.write(utils.to_utf8("%s %s\n" % self.syn0.shape))
            # store in sorted order: most frequent words at the top
            for word, vocab in sorted(iteritems(self.vocab), key=lambda item:-item[1].count):
                row = self.syn0[vocab.index]
                if binary:
                    fout.write(utils.to_utf8(word) + b" " + row.tostring())
                else:
                    fout.write(utils.to_utf8("%s %s\n" % (word, ' '.join("%f" % val for val in row))))


    @classmethod
    def load_word2vec_format(cls, fname, fvocab=None, binary=False, norm_only=True):
        """
        Load the input-hidden weight matrix from the original C word2vec-tool format.

        Note that the information stored in the file is incomplete (the binary tree is missing),
        so while you can query for word similarity etc., you cannot continue training
        with a model loaded this way.

        `binary` is a boolean indicating whether the data is in binary word2vec format.
        `norm_only` is a boolean indicating whether to only store normalised word2vec vectors in memory.
        Word counts are read from `fvocab` filename, if set (this is the file generated
        by `-save-vocab` flag of the original C tool).
        """
        counts = None
        if fvocab is not None:
            logger.info("loading word counts from %s" % (fvocab))
            counts = {}
            with utils.smart_open(fvocab) as fin:
                for line in fin:
                    word, count = utils.to_unicode(line).strip().split()
                    counts[word] = int(count)

        logger.info("loading projection weights from %s" % (fname))
        with utils.smart_open(fname) as fin:
            header = utils.to_unicode(fin.readline())
            vocab_size, layer1_size = map(int, header.split())  # throws for invalid file format
            result = Graph2Vec(size=layer1_size)
            result.syn0 = zeros((vocab_size, layer1_size), dtype=REAL)
            if binary:
                binary_len = dtype(REAL).itemsize * layer1_size
                for line_no in xrange(vocab_size):
                    # mixed text and binary: read text first, then binary
                    word = []
                    while True:
                        ch = fin.read(1)
                        if ch == b' ':
                            break
                        if ch != b'\n':  # ignore newlines in front of words (some binary files have newline, some don't)
                            word.append(ch)
                    word = utils.to_unicode(b''.join(word))
                    if counts is None:
                        result.vocab[word] = Vocab(index=line_no, count=vocab_size - line_no)
                    elif word in counts:
                        result.vocab[word] = Vocab(index=line_no, count=counts[word])
                    else:
                        logger.warning("vocabulary file is incomplete")
                        result.vocab[word] = Vocab(index=line_no, count=None)
                    result.index2word.append(word)
                    result.syn0[line_no] = fromstring(fin.read(binary_len), dtype=REAL)
            else:
                for line_no, line in enumerate(fin):
                    parts = utils.to_unicode(line).split()
                    if len(parts) != layer1_size + 1:
                        raise ValueError("invalid vector on line %s (is this really the text format?)" % (line_no))
                    word, weights = parts[0], map(REAL, parts[1:])
                    if counts is None:
                        result.vocab[word] = Vocab(index=line_no, count=vocab_size - line_no)
                    elif word in counts:
                        result.vocab[word] = Vocab(index=line_no, count=counts[word])
                    else:
                        logger.warning("vocabulary file is incomplete")
                        result.vocab[word] = Vocab(index=line_no, count=None)
                    result.index2word.append(word)
                    result.syn0[line_no] = weights
        logger.info("loaded %s matrix from %s" % (result.syn0.shape, fname))
        result.init_sims(norm_only)
        return result


    def most_similar(self, positive=[], negative=[], topn=10):
        """
        Find the top-N most similar words. Positive words contribute positively towards the
        similarity, negative words negatively.

        This method computes cosine similarity between a simple mean of the projection
        weight vectors of the given words, and corresponds to the `word-analogy` and
        `distance` scripts in the original word2vec implementation.

        Example::

          >>> trained_model.most_similar(positive=['woman', 'king'], negative=['man'])
          [('queen', 0.50882536), ...]

        """
        self.init_sims()

        if isinstance(positive, string_types) and not negative:
            # allow calls like most_similar('dog'), as a shorthand for most_similar(['dog'])
            positive = [positive]

        # add weights for each word, if not already present; default to 1.0 for positive and -1.0 for negative words
        positive = [(word, 1.0) if isinstance(word, string_types + (ndarray,))
                                else word for word in positive]
        negative = [(word, -1.0) if isinstance(word, string_types + (ndarray,))
                                 else word for word in negative]

        # compute the weighted average of all words
        all_words, mean = set(), []
        for word, weight in positive + negative:
            if isinstance(word, ndarray):
                mean.append(weight * word)
            elif word in self.vocab:
                mean.append(weight * self.syn0norm[self.vocab[word].index])
                all_words.add(self.vocab[word].index)
            else:
                raise KeyError("word '%s' not in vocabulary" % word)
        if not mean:
            raise ValueError("cannot compute similarity with no input")
        mean = matutils.unitvec(array(mean).mean(axis=0)).astype(REAL)

        dists = dot(self.syn0norm, mean)
        if not topn:
            return dists
        best = argsort(dists)[::-1][:topn + len(all_words)]
        # ignore (don't return) words from the input
        result = [(self.index2word[sim], float(dists[sim])) for sim in best if sim not in all_words]
        return result[:topn]


    def __getitem__(self, word):
        """
        Return a word's representations in vector space, as a 1D numpy array.

        Example::

          >>> trained_model['woman']
          array([ -1.40128313e-02, ...]

        """
        return self.syn0[self.vocab[word].index]


    def __contains__(self, word):
        return word in self.vocab



    def init_sims(self, replace=False):
        """
        Precompute L2-normalized vectors.

        If `replace` is set, forget the original vectors and only keep the normalized
        ones = saves lots of memory!

        Note that you **cannot continue training** after doing a replace. The model becomes
        effectively read-only = you can call `most_similar`, `similarity` etc., but not `train`.

        """
        if getattr(self, 'syn0norm', None) is None or replace:
            logger.info("precomputing L2-norms of word weight vectors")
            if replace:
                for i in xrange(self.syn0.shape[0]):
                    self.syn0[i, :] /= sqrt((self.syn0[i, :] ** 2).sum(-1))
                self.syn0norm = self.syn0
                if hasattr(self, 'syn1'):
                    del self.syn1
            else:
                self.syn0norm = (self.syn0 / sqrt((self.syn0 ** 2).sum(-1))[..., newaxis]).astype(REAL)



if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s : %(threadName)s : %(levelname)s : %(message)s', level=logging.INFO)
    logging.info("running %s" % " ".join(sys.argv))
    logging.info("using optimization %s" % FAST_VERSION)
    program = os.path.basename(sys.argv[0])
    seterr(all='raise')  # don't ignore numpy errors

    graph_file = 'E:/workspace/jarvis/data/TCBB/forDep2vec_PPI_10t_input_new'
    model = Graph2Vec(graph_file, size=400, workers=1, iter=1)
    model.save_word2vec_format('E:/data/wordVectors/GraphVector_new_10t_d400.txt', binary=False)
    logging.info("finished running %s" % program)
