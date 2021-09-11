# Webdriver performs all the web mechanics
# - choose a browser, and link it to the browser's driver
from selenium import webdriver
from bs4 import BeautifulSoup
from collections import OrderedDict
import numpy as np
from spacy.lang.en.stop_words import STOP_WORDS
import spacy
from sklearn.feature_extraction.text import CountVectorizer
import pt_core_news_sm

nlp = spacy.load('en_core_web_sm')
nlp2 = pt_core_news_sm.load()

# Initialize selenium to extract page data

PATH = "C:\Program Files (x86)\chromedriver.exe"

driver = webdriver.Chrome(PATH)
driver.get("https://www.unicef.org/press-releases/joint-covax-statement-supply-forecast-2021-and-early-2022");

soup = BeautifulSoup(driver.page_source, "lxml")
get_info = soup.findAll('div', {"class": "field_component_text_content"})

info = []
for i in range(0, len(get_info[0])):
    try:
        info.append(get_info[0].contents[i].getText())
    except:
        pass


text = "".join(info)

class TextRank4Keyword():
    """Extract keywords from text"""
    
    def __init__(self):
        self.d = 0.85 # damping coefficient, usually is .85
        self.min_diff = 1e-5 # convergence threshold
        self.steps = 10 # iteration steps
        self.node_weight = None # save keywords and its weight

    
    def set_stopwords(self, stopwords):  
        """Set stop words"""
        for word in STOP_WORDS.union(set(stopwords)):
            lexeme = nlp.vocab[word]
            lexeme.is_stop = True
    
    def sentence_segment(self, doc, candidate_pos, lower):
        """Store those words only in cadidate_pos"""
        sentences = []
        for sent in doc.sents:
            selected_words = []
            for token in sent:
                # Store words only with cadidate POS tag
                if token.pos_ in candidate_pos and token.is_stop is False:
                    if lower is True:
                        selected_words.append(token.text.lower())
                    else:
                        selected_words.append(token.text)
            sentences.append(selected_words)
        return sentences
        
    def get_vocab(self, sentences):
        """Get all tokens"""
        vocab = OrderedDict()
        i = 0
        for sentence in sentences:
            for word in sentence:
                if word not in vocab:
                    vocab[word] = i
                    i += 1
        return vocab
    
    def get_token_pairs(self, window_size, sentences):
        """Build token_pairs from windows in sentences"""
        token_pairs = list()
        for sentence in sentences:
            for i, word in enumerate(sentence):
                for j in range(i+1, i+window_size):
                    if j >= len(sentence):
                        break
                    pair = (word, sentence[j])
                    if pair not in token_pairs:
                        token_pairs.append(pair)
        return token_pairs
        
    def symmetrize(self, a):
        return a + a.T - np.diag(a.diagonal())
    
    def get_matrix(self, vocab, token_pairs):
        """Get normalized matrix"""
        # Build matrix
        vocab_size = len(vocab)
        g = np.zeros((vocab_size, vocab_size), dtype='float')
        for word1, word2 in token_pairs:
            i, j = vocab[word1], vocab[word2]
            g[i][j] = 1
            
        # Get Symmeric matrix
        g = self.symmetrize(g)
        
        # Normalize matrix by column
        norm = np.sum(g, axis=0)
        g_norm = np.divide(g, norm, where=norm!=0) # this is ignore the 0 element in norm
        
        return g_norm

    
    def get_keywords(self, number=10):
        """Print top number keywords"""
        node_weight = OrderedDict(sorted(self.node_weight.items(), key=lambda t: t[1], reverse=True))
        for i, (key, value) in enumerate(node_weight.items()):
            print(key + ' - ' + str(value))
            if i > number:
                break
        
        
    def analyze(self, text, 
                candidate_pos=['NOUN', 'PROPN'], 
                window_size=4, lower=False, stopwords=list()):
        """Main function to analyze text"""
        
        # Set stop words
        self.set_stopwords(stopwords)
        
        # Pare text by spaCy
        doc = nlp(text)
        
        # Filter sentences
        sentences = self.sentence_segment(doc, candidate_pos, lower) # list of list of words
        
        # Build vocabulary
        vocab = self.get_vocab(sentences)
        
        # Get token_pairs from windows
        token_pairs = self.get_token_pairs(window_size, sentences)
        
        # Get normalized matrix
        g = self.get_matrix(vocab, token_pairs)
        
        # Initionlization for weight(pagerank value)
        pr = np.array([1] * len(vocab))
        
        # Iteration
        previous_pr = 0
        for epoch in range(self.steps):
            pr = (1-self.d) + self.d * np.dot(g, pr)
            if abs(previous_pr - sum(pr))  < self.min_diff:
                break
            else:
                previous_pr = sum(pr)

        # Get weight for each node
        node_weight = dict()
        for word, index in vocab.items():
            node_weight[word] = pr[index]
        
        self.node_weight = node_weight

tr4w = TextRank4Keyword()
tr4w.analyze(text, candidate_pos = ['NOUN', 'PROPN'], window_size=4, lower=False)
tr4w.get_keywords(10)

### Summarising

doc = nlp2(text)
corpus = [sent.text.lower() for sent in doc.sents ]
cv = CountVectorizer(stop_words=list(STOP_WORDS))   
cv_fit=cv.fit_transform(corpus)    
word_list = cv.get_feature_names();    
count_list = cv_fit.toarray().sum(axis=0)
word_frequency = dict(zip(word_list,count_list))

val=sorted(word_frequency.values())
higher_word_frequencies = [word for word,freq in word_frequency.items() if freq in val[-3:]]
print("\nWords with higher frequencies: ", higher_word_frequencies)

# gets relative frequency of words
higher_frequency = val[-1]
for word in word_frequency.keys():  
    word_frequency[word] = (word_frequency[word]/higher_frequency)

sentence_rank={}
for sent in doc.sents:
    for word in sent :       
        if word.text.lower() in word_frequency.keys():            
            if sent in sentence_rank.keys():
                sentence_rank[sent]+=word_frequency[word.text.lower()]
            else:
                sentence_rank[sent]=word_frequency[word.text.lower()]
top_sentences=(sorted(sentence_rank.values())[::-1])
top_sent=top_sentences[:3]

summary=[]
for sent,strength in sentence_rank.items():  
    if strength in top_sent:
        summary.append(sent)
    else:
        continue
for i in summary:
    print(i,end=" ")