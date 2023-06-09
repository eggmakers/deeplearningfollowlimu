import collections
import re
from d2l import torch as d2l


#将数据集读取到有多条文本行组成的列表中
d2l.DATA_HUB['time_machine'] = (d2l.DATA_URL + 'timemachine.txt',
                                '090b5e7e70c295757f55df93cb0a180b9691891a')

def read_time_machine():
    """Load the time machine dataset into a list of text lines"""
    with open(d2l.download('time_machine'), 'r') as f:
        lines = f.readlines()
    return [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines]

lines = read_time_machine()
print(f'# 文本总行数: {len(lines)}')
print(lines[0])
print(lines[10])


#每个文本序列有被拆分成一个标记列表
def tokenize(lines, token='word'):
    """将文本拆分为单词或字符标记"""
    if token == 'word':
        return [line.split() for line in lines]
    elif token == 'char':
        return [list(line) for line in lines]
    else:
        print("error:unknown token:" + token)

tokens = tokenize(lines)
for i in range(11):
    print(tokens[i])


#构建一个字典，通常也叫词汇表，用来将字符串类型的标记映射到从0开始的数字索引中
class Vocab:
    """文本词汇表"""
    def __init__(self, tokens=None, min_freq=0, reserved_torkens=None):
        if tokens is None:
            tokens = []
        if reserved_torkens is None:
            reserved_torkens = []
        counter = count_corpus(tokens)
        self._token_freqs = sorted(counter.items(), key=lambda x: x[1],
                                   reverse=True)
        #未知词源的索引为0
        self.idx_to_token = ['<unk>'] + reserved_torkens
        self.token_to_idx = {token: idx
                             for idx, token in enumerate(self.idx_to_token)}
        for token,freq in self._token_freqs:
            if freq < min_freq:
                break
            if token not in self.token_to_idx:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1
    
    def __len__(self):
        return len(self.idx_to_token)
    
    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]
        
    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]
    
    @property
    def unk(self):
        return 0
    
    @property
    def token_freqs(self):
        return self._token_freqs
    
def count_corpus(tokens):  #@save
    """统计词元的频率"""
    # 这里的tokens是1D列表或2D列表
    if len(tokens) == 0 or isinstance(tokens[0], list):
        # 将词元列表展平成一个列表
        tokens = [token for line in tokens for token in line]
    return collections.Counter(tokens)


#构建词汇表
vocab = Vocab(tokens)
print(list(vocab.token_to_idx.items())[:10])


#将文本转换成数字索引
for i in [0, 10]:
    print('words:', tokens[i])
    print('indices:', vocab[tokens[i]])

def load_corpus_time_machine(max_tokens=-1):
    """返回时光机器数据集的词源索引列表和词表"""
    lines = read_time_machine()
    tokens = tokenize(lines, 'char')
    vocab = Vocab(tokens)
    corpus = [vocab[token] for line in tokens for token in line]
    if max_tokens > 0:
        corpus = corpus[:max_tokens]
    return corpus, vocab

corpus, vocab = load_corpus_time_machine()
print(len(corpus), len(vocab))