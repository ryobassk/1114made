class Vocab(object):
    #イニシャライザ
    def __init__(self):
        self.w2i = {} #単語⇒IDの辞書
        self.i2w = {} #ID⇒単語の辞書
        
        #特殊文字（0:パッティング文字、1:開始文字等、2:終了文字、3:未知の文字）
        self.special_chars = ['<pad>', '<s>', '</s>', '<unk>']
        self.bos_char = self.special_chars[1]
        self.eos_char = self.special_chars[2]
        self.oov_char = self.special_chars[3]

    #ファイルの読み込みと辞書の作成
    def fit(self, path):
        self._words = set()
        
        #文章を保存（'彼 は 走 る の が とても 早 い 。', ）
        with open(path, 'r', encoding="utf-8") as f:
            sentences = f.read().splitlines() 
            
        #文章を単語ごとに保存（'彼', 'は',・・・・・・ ）
        for sentence in sentences:
            self._words.update(sentence.split())
        
        #単語⇒IDの辞書の作成(4～8777)
        self.w2i = {w: (i + len(self.special_chars))
                    for i, w in enumerate(self._words)}
        
        #単語⇒IDの辞書の作成(0～3)特殊文字、開始記号等
        for i, w in enumerate(self.special_chars):
            self.w2i[w] = i
            
        #ID⇒単語の辞書の作成   
        self.i2w = {i: w for w, i in self.w2i.items()}
        
        #辞書の出力
        import datetime
        import pathlib
        p_file = pathlib.Path(path)
        now = datetime.datetime.now()
        fdict_id = open('./data/sakusei/datamake/'+str(now.month)+str(now.day)+'dict_id_'+p_file.stem+'.txt', 'w')
        for key,atai in self.i2w.items():
            fdict_id.write(str(key)+'\t'+str(atai)+'\n')
        
    #ファイルの読み込み、全体をIDに変換
    def transform(self, path, bos=False, eos=False):
        output = []
        
        #文章を保存（'彼 は 走 る の が とても 早 い 。', ）
        with open(path, 'r', encoding="utf-8") as f:
            sentences = f.read().splitlines()

        for sentence in sentences:
            sentence = sentence.split()
            if bos:
                sentence = [self.bos_char] + sentence
            if eos:
                sentence = sentence + [self.eos_char]
            output.append(self.encode(sentence))

        return output
        
        
    #単語をIDに変換
    def encode(self, sentence):
        output = []
        
        #1文章を入力,wは単語
        for w in sentence:
            #辞書にない単語は未知語としてIDを振る
            if w not in self.w2i:
                idx = self.w2i[self.oov_char]
            #単語を辞書を用いてIDに変換
            else:
                idx = self.w2i[w]
            output.append(idx)
        
        return output
    
    #IDを単語に変換
    def decode(self, sentence):
        return [self.i2w[id] for id in sentence]
