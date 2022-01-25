# this code contains all the costum function used in main.inpy
from libs import *
def expand_contractions(tweet):
    return contractions.fix(tweet)

def and_fixer(tweet):
    return re.sub(r"&", " and ", tweet)

def punch_fixer(tweet):
    return re.sub(r"(\.|\?|;|:|!)+", ". ", tweet)

def seperator_fixer(tweet):
    return re.sub(r"(\$|,|/|<|>|=|\+|%|\n)+", " ", tweet)

def line_fixer(tweet):
    return re.sub(r"(-|_)+", " ", tweet)

def number_fixer(tweet):
    return re.sub(r"[0-9]+", " ", tweet)

def space_fixer(tweet):
    return re.sub(r" +", " ", tweet)

##### should be done after contraction fixes
def direct_speech_fixer(tweet):
    return re.sub(r"(\"|\(|\)|')+", " ", tweet)

def dot_fixer(tweet):
    try:
        if not tweet[-1] == ".":
            tweet += "."
    except:
        pass
    tweet = re.sub(r"( |\s)+\. ", ". ", tweet)
    tweet = re.sub(r"( |\s)+\.", ".", tweet)
    tweet = re.sub(r"\.+ ", ". ", tweet)
    tweet = re.sub(r"\.+", ".", tweet)
    tweet = re.sub(r"\. \.", ".", tweet)
    tweet =  re.sub(r" \. ", ". ", tweet)
    return tweet

def insult_fixer(tweet):
    return re.sub(r"[A-Za-z]*\*+[A-Za-z]*", "fohsh", tweet)

def single_alph_fixer(tweet):
    return re.sub(r" [A-Za-z] ", "", tweet)

def sentence_parser(tweet):
    # tweet = "<s> " + tweet
    tweet = re.sub(r"\.$", " </s>", tweet)
    tweet = "<s> " + tweet
    tweet = re.sub(r"\. ", " </s> <s> ", tweet)
    return tweet


def costum_preprocessing(tweet):
    tweet = expand_contractions(tweet)
    tweet = p.clean(tweet)
    tweet = and_fixer(tweet)
    tweet = punch_fixer(tweet)
    tweet = seperator_fixer(tweet)
    tweet = line_fixer(tweet)
    tweet = number_fixer(tweet)
    tweet = space_fixer(tweet)
    tweet = direct_speech_fixer(tweet)
    tweet = dot_fixer(tweet)
    tweet = insult_fixer(tweet)
    tweet = single_alph_fixer(tweet)
    return tweet

def lower(tweet):
    return tweet.lower() 
def lemma(tweet):
    lemmatizer = WordNetLemmatizer()
    sep = tweet.split()
    res = ""
    for i, word in enumerate(sep):
        res += lemmatizer.lemmatize(word)
        if not i == (len(sep)-1):
            res += " "
    return res

def stem(tweet):
    ps = PorterStemmer()
    sep = tweet.split()
    res = ""
    for i, word in enumerate(sep):
        res += ps.stem(word)
        if not i == (len(sep)-1):
            res += " "
    return res

#Frequency of words
def plot_wordcloud(dataset, desc):
    stop_words = set(stopwords.words('english'))
    stop_words.add("<s>")
    stop_words.add("</s>")
    all_words = []
    for tweet_list in dataset.text:
        all_words += tweet_list

    all_words_cleaned = []
    for word in all_words:
        word = word.lower()
        if not word in stop_words:
            all_words_cleaned.append(word)
    fdist = FreqDist(all_words_cleaned)
    #WordCloud
    wc = WordCloud(width=800, height=400, max_words=50).generate_from_frequencies(fdist)
    plt.figure(figsize=(12,10))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.savefig('./Res/Clouds/'+desc+".png")
    plt.title(desc)
    plt.show()

def prep(corpus, funcs, tnkz_func):
    '''
    corpus should be a csv file which has a text column
    each row should be a string
    '''
    data = copy.deepcopy(corpus)
    # selectable preprocessing steps
    for func in funcs:
        data['text'] = data['text'].apply(func)
    
    # parse sentences
    data['text'] = data['text'].apply(sentence_parser)
    
    # tokenize text _ slectable
    data['text'] = data['text'].apply(tnkz_func)
    return data

def word_tokenizer(tweet):
    tknzr = TweetTokenizer()
    return tknzr.tokenize(tweet)

def char_tokenizer(tweet):
    specials = ['<', '>', 's', '/']
    tokenized = []
    for i, char in enumerate(tweet):
        if char in specials:
            if char == specials[0]:
                if tweet[i:i+3] == "<s>":
                    tokenized.append('<s>')
                    continue
                elif tweet[i:i+4] == "</s>":
                    tokenized.append('</s>')
                    continue
            else: continue
        tokenized.append(char)
    return tokenized

def train_models(data, LM):
    '''
    data.labels should be 0s and 1s
    '''
    train_1 = list(data[data.label==1].text)
    train_0 = list(data[data.label==0].text)
    
    # creat grams and vocabs
    # lazy mode, so there is no mem cost at this point
    train_1_0, vocab_1_0 = padded_everygram_pipeline(1, train_0)
    train_1_1, vocab_1_1 = padded_everygram_pipeline(1, train_1)
    # train_2_0, vocab_2_0 = padded_everygram_pipeline(2, train_0)
    # train_2_1, vocab_2_1 = padded_everygram_pipeline(2, train_1)

    
    # create models
    lm_1_0 = LM(order=1)
    lm_1_1 = LM(order=1)
    # lm_2_0 = LM(order=2)
    # lm_2_1 = LM(order=2)

    
    # fit models
    lm_1_0.fit(train_1_0, vocab_1_0)
    lm_1_1.fit(train_1_1, vocab_1_1)
    # lm_2_0.fit(train_2_0, vocab_2_0)
    # lm_2_1.fit(train_2_1, vocab_2_1)
    
    # return lm_1_0, lm_1_1, lm_2_0, lm_2_1
    return lm_1_0, lm_1_1, None, None

def predict_labels(data, lm_0, lm_1, order):
    labels = []
    for sent in data.text:
        try:
            p0 = lm_0.perplexity(ngrams(sent, order))
        except:
            labels.append(1)
            continue
        try:
            p1 = lm_1.perplexity(ngrams(sent, order))
        except:
            labels.append(0)
            continue

        if  p0 < p1:
            labels.append(0)
        else:
            labels.append(1)
    return labels

def plot_confusion_matrix(true_labels, pred_labels, desc):
    cm = confusion_matrix(true_labels, pred_labels, labels=[0, 1], normalize='true')
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=[0, 1])
    disp.plot(include_values=True,cmap='viridis', ax=None, xticks_rotation='horizontal',values_format=None)
    plt.title(desc)
    plt.savefig('./Res/CMs/'+desc+".png")
    plt.close('all')
    # plt.show()


def plot_ROC(true_labels, pred_labels, desc):
    fpr, tpr, thresholds = roc_curve(true_labels, pred_labels)
    roc_auc = auc(fpr, tpr)
    display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name=" ")
    display.plot()
    plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')
    plt.grid()
    plt.title(desc)
    plt.savefig('./Res/ROC/'+desc+".png")
    plt.close('all')
    # plt.show()  

def evaluate_model(true_labels, pred_labels, discription=" "):
    # print("_"*100)
    print(discription)
    plot_confusion_matrix(true_labels, pred_labels, discription)
    plot_ROC(true_labels, pred_labels, discription)

    acc = accuracy_score(true_labels, pred_labels)
    prec = precision_score(true_labels, pred_labels)
    recall = recall_score(true_labels, pred_labels)
    f1_macro = f1_score(true_labels, pred_labels, average="macro")
    f1_micro = f1_score(true_labels, pred_labels, average="micro")

    # print(f'accuracy score  = {acc}')
    # print(f'precision score = {prec}')
    # print(f'recall score    = {recall}')
    # print(f'f1 macro score  = {f1_macro}')
    # print(f'f1 micro score  = {f1_micro}')
    # print("_"*100)
    return [discription, acc, prec, recall, f1_macro, f1_micro]

def create_masks():
    masks = []
    for i in [False, True]:
        for j in [False, True]:
            masks.append([i, j])
    return masks

def create_desc(tknz_name, model_name, opt_names, order, mode):
    delim = "|"
    if order == 1:
        desc = "|unigram|"+mode+delim
    else:
        desc = "|bigram|"+mode+delim
    desc += (tknz_name+delim)
    desc += (model_name+delim)
    desc += ('basic prep'+delim)
    for opt_name in opt_names:
        desc +=(opt_name+delim)
    return desc

def create_df_from_records(records, file_name):
    combinations = []
    accs = []
    precs = []
    recalls = []
    f1macros = []
    f1micros = []
    for rec in records:
        combinations.append(rec[0])
        accs.append(rec[1])
        precs.append(rec[2])
        recalls.append(rec[3])
        f1macros.append(rec[4])
        f1micros.append(rec[5])
    d = {"model":combinations, "f1-macro":f1macros, "f1-micro":f1micros , "acc":accs, "precision":precs, "recall":recalls}
    df = pd.DataFrame(d)
    df.to_csv('./Res/'+file_name+'.csv')
    return df

def load_data(hate_train_path, hate_dev_path, hate_test_path, off_train_path):
    hate_train = pd.read_csv(hate_train_path)
    plt.hist(hate_train.label);
    plt.xlabel('label')
    plt.ylabel('count')
    plt.title('HateEval/train classes')
    plt.grid()
    plt.savefig('./Res/Data_stats/HateTrain.jpg')
    plt.close('all')
    
    hate_dev = pd.read_csv(hate_dev_path)
    plt.hist(hate_dev.label)
    plt.xlabel('label')
    plt.ylabel('count')
    plt.title('HateEval/dev classes')
    plt.grid()
    plt.savefig('./Res/Data_stats/HateDev.jpg')
    plt.close('all')
    
    off_train = pd.read_csv(off_train_path)
    plt.hist(off_train.label)
    plt.xlabel('label')
    plt.ylabel('count')
    plt.title('OffenseEval/train classes')
    plt.grid()
    plt.savefig('./Res/Data_stats/OffTrain.jpg')
    plt.close('all')
    
    hate_test = pd.read_csv(hate_test_path)
    
    return hate_train, hate_dev, hate_test, off_train
