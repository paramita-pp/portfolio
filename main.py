# name Paramita Parinyanupap n10597123
# import important library
import os, glob
import string
import nltk
from nltk.corpus import stopwords
from stemming.porter2 import stem
import math
import coll
import df

# Part 1 Automatic Training
# Q1 and Q2
# extract query from topic
def extract_query():
    query_file = open('Topic_definitions.txt', 'r')
    a = query_file.readlines()
    topics = [topic for topic in a if '<num> Number' in topic]
    topics = [topic.replace('<num> Number: ', '') for topic in topics]
    topics = [topic.replace('\n', '') for topic in topics]
    topics = [topic.strip() for topic in topics]
    titles = [title for title in a if '<title>' in title]
    titles = [title.replace('<title>', '') for title in titles]
    titles = [title.replace('\n', '') for title in titles]
    titles = [title.translate(str.maketrans('','', string.punctuation)) for title in titles]
    titles = [title.strip() for title in titles]
    query_file.close()
    topics_dict = {}
    for i in range(len(topics)):
        topics_dict[topics[i]] = titles[i]
    return topics_dict

def avg_doc_len(coll):
    tot_dl = 0
    for id, doc in coll.get_docs().items():
        tot_dl += doc.get_doc_len()
    return tot_dl/coll.get_num_docs()

def bm25(coll, q, df):
    bm25s = {}
    avg_dl = avg_doc_len(coll)
    no_docs = coll.get_num_docs()
    for id, doc in coll.get_docs().items():
        query_terms = q.split()
        qfs = {}
        for t in query_terms:
            term = stem(t.lower())
            try:
                qfs[term] +=1
            except KeyError:
                qfs[term] = 1
        k = 1.2 * ((1-0.75) +0.75 *(doc.get_doc_len()/float(avg_dl)))
        bm25_ =0.0
        for qt in qfs.keys():
            n = 0
            if qt in df.keys():
                n = df[qt]
                f = doc.get_term_count(qt)
                qf = qfs[qt]
                bm = math.log(1.0 / ((n + 0.5) / (no_docs - n + 0.5)), 2) * (((1.2 + 1) * f) / (k + f)) * ( ((100 + 1) * qf) / float(100 + qf))
                bm25_ += bm
        bm25s[doc.get_docid()] = bm25_
    return bm25s
def isrelevant(topic, coll_):
    a = int(topic[1:])
    df_ = df.calc_df(coll_)
    bm25_1 = bm25(coll_, query_dict[topic], df_)
    os.chdir('..')
    b = a - 100
    # model_name = 'BaselineModel_' + topic + '.dat'
    model_name = 'B_Result' + str(b) + '.dat'
    wFile = open(model_name, 'a')
    for (k, v) in sorted(bm25_1.items(), key=lambda x: x[1], reverse=True):
        wFile.write(k + ' ' + str(v) + '\n')
    wFile.close()

    filename = 'PTraining_benchmark_' + topic + '.txt'
    writeFile = open(filename, 'a')
    # datname = 'BaselineModel_' + topic + '.dat'
    datname = 'B_Result'+ str(b) + '.dat'
    datFile = open(datname)
    file_ = datFile.readlines()
    for line in file_:
        line = line.strip()
        lineStr = line.split()
        if float(lineStr[1]) > 1.0:
            writeFile.write(topic + ' ' + lineStr[0] + ' 1' + '\n')
        else:
            writeFile.write(topic + ' ' + lineStr[0] + ' 0' + '\n')
    writeFile.close()
    datFile.close()



# Part 2 Information Filtering model include both training and testing algorithms
# Q4 and Q5
# training algorithm
def training(coll, topic):
    benFilename = 'PTraining_benchmark_R' + topic + '.txt'
    benFile = open(benFilename)
    file_ = benFile.readlines()
    ben = {}
    for line in file_:
        line = line.strip()
        lineList = line.split()
        ben[lineList[1]] = float(lineList[2])
    benFile.close()
    theta = 3.5

    T = {}
    # select T from positive documents and r(tk)
    for id, doc in coll.get_docs().items():
        if ben[id] > 0:
            for term, freq in doc.terms.items():
                try:
                    T[term] += 1
                except KeyError:
                    T[term] = 1
    # calculate n(tk)
    ntk = {}
    for id, doc in coll.get_docs().items():
        for term in doc.get_term_list():
            try:
                ntk[term] += 1
            except KeyError:
                ntk[term] = 1

    # calculate N and R

    No_docs = coll.get_num_docs()
    R = 0
    for id, fre in ben.items():
        if ben[id] > 0:
            R += 1

    for id, rtk in T.items():
        T[id] = ((rtk + 0.5) / (R - rtk + 0.5)) / ((ntk[id] - rtk + 0.5) / (No_docs - ntk[id] - R + rtk + 0.5))

        # calculate the mean of w4 weights.
    meanW4 = 0
    for id, rtk in T.items():
        meanW4 += rtk
    meanW4 = meanW4 / len(T)

    # Features selection
    Features = {t: r for t, r in T.items() if r > meanW4 + theta}
    return Features

# testing algorithm
def testing(coll, features):
    ranks = {}
    for id, doc in coll.get_docs().items():
        Rank = 0
        for term in features.keys():
            if term in doc.get_term_list():
                try:
                    ranks[id] += features[term]
                except KeyError:
                    ranks[id] = features[term]
    return ranks

def writeModel_w4(bm25_weight, topic):
    filename = 'Model_w4_' + topic + '.dat'
    wFile = open(filename, 'w')
    for (k,v) in sorted(bm25_weight.items(), key = lambda x: x[1], reverse=True):
        wFile.write(k + ' ' + str(v) + '\n')
    featureFile = open(filename)
    file_ = featureFile.readlines()
    features = {}
    for line in file_:
        line = line.strip()
        lineList = line.split()
        features[lineList[0]] = float(lineList[1])
    featureFile.close()

    ranks = testing(coll_, features)
    a = topic[1:]
    filename2 = 'IF_Result' + a + '.dat'
    wFile = open(filename2, 'w')
    for (d, v) in sorted(ranks.items(), key = lambda x: x[1], reverse = True):
        wFile.write(d + ' ' + str(v) + '\n')
    wFile.close()

# Part 3 Testing
# Q6
def task1(file1, file2):
    A = {}
    B = {}
    for line in open(file1):
        line = line.strip()
        line1 = line.split()
        A[line1[1]] = int(float(line1[2]))
    for line in open(file2):
        line = line.strip()
        line1 = line.split()
        B[line1[1]] = int(float(line1[2]))
    return (A,B)

def eval(topic, IF_result, training_file):
    # return output topic, precision, recall, F1 in csv file
    (rel_doc, retrived_doc) =task1(IF_result, training_file)
    R = 0
    for (x,y) in rel_doc.items():
        if (y == 1):
            R = R+1
    print('the number of relevant docs: ' + str(R))

    R1 = 0
    for (x,y) in retrived_doc.items():
        if (y == 1):
            R1 +=1
    print('the number of retrieved docs: ' + str(R1))

    RR1 = 0
    for (x,y) in rel_doc.items():
        if ( y == 1) & (retrived_doc[x] == 1):
            RR1+= 1
    print('the number of retrieved docs that are relevant: ' + str(RR1))

    r = float(RR1)/float(R)
    p = float(RR1) /float(R1)
    F1 = 2 *p*r/(p+r)
    # print('recall = ' + str(r))
    # print('precision = ' + str(p))
    # print('F-Measure = ' + str(F1))
    topic = int(topic) - 100
    filename = 'EResult' + str(topic) + '.dat'
    wFile = open(filename, 'w')
    wFile.write('Topic Precision Recall F1\n')
    wFile.write(str(topic) + ' ' + str(r) + ' ' + str(p) + ' ' + str(F1))

def ttest():
    pass



if __name__ == '__main__':
    # get query from topic defition
    query_dict = extract_query()
    stop_words = stopwords.words('english')

    all_coll = {}
    os.chdir('dataset101-150')
    for coll_fname in glob.glob('Training*'):
        topic = 'R' + str(coll_fname[8:])
        coll_ = coll.parse_rcv_coll(coll_fname, stop_words)
        # q1,q2,q3
        isrelevant(topic, coll_)

    coll_101 = coll.parse_rcv_coll('Training101', stop_words)
    os.chdir('..')
    coll_102 = coll.parse_rcv_coll('Training102', stop_words)
    os.chdir('..')
    coll_103 = coll.parse_rcv_coll('Training103', stop_words)
    os.chdir('..')
    coll_104 = coll.parse_rcv_coll('Training104', stop_words)
    os.chdir('..')
    coll_105 = coll.parse_rcv_coll('Training105', stop_words)
    os.chdir('..')
    coll_106 = coll.parse_rcv_coll('Training106', stop_words)
    os.chdir('..')
    coll_107 = coll.parse_rcv_coll('Training107', stop_words)
    os.chdir('..')
    coll_108 = coll.parse_rcv_coll('Training108', stop_words)
    os.chdir('..')
    coll_109 = coll.parse_rcv_coll('Training109', stop_words)
    os.chdir('..')
    coll_110 = coll.parse_rcv_coll('Training110', stop_words)
    os.chdir('..')
    coll_111 = coll.parse_rcv_coll('Training111', stop_words)
    os.chdir('..')
    coll_112 = coll.parse_rcv_coll('Training112', stop_words)
    os.chdir('..')
    coll_113 = coll.parse_rcv_coll('Training113', stop_words)
    os.chdir('..')
    coll_114 = coll.parse_rcv_coll('Training114', stop_words)
    os.chdir('..')
    coll_115 = coll.parse_rcv_coll('Training115', stop_words)
    os.chdir('..')
    coll_116 = coll.parse_rcv_coll('Training116', stop_words)
    os.chdir('..')
    coll_117 = coll.parse_rcv_coll('Training117', stop_words)
    os.chdir('..')
    coll_118 = coll.parse_rcv_coll('Training118', stop_words)
    os.chdir('..')
    coll_119 = coll.parse_rcv_coll('Training119', stop_words)
    os.chdir('..')
    coll_120 = coll.parse_rcv_coll('Training120', stop_words)
    os.chdir('..')
    coll_121 = coll.parse_rcv_coll('Training121', stop_words)
    os.chdir('..')
    coll_122 = coll.parse_rcv_coll('Training122', stop_words)
    os.chdir('..')
    coll_123 = coll.parse_rcv_coll('Training123', stop_words)
    os.chdir('..')
    coll_124 = coll.parse_rcv_coll('Training124', stop_words)
    os.chdir('..')
    coll_125 = coll.parse_rcv_coll('Training125', stop_words)
    os.chdir('..')
    coll_126 = coll.parse_rcv_coll('Training126', stop_words)
    os.chdir('..')
    coll_127 = coll.parse_rcv_coll('Training127', stop_words)
    os.chdir('..')
    coll_128 = coll.parse_rcv_coll('Training128', stop_words)
    os.chdir('..')
    coll_129 = coll.parse_rcv_coll('Training129', stop_words)
    os.chdir('..')
    coll_130 = coll.parse_rcv_coll('Training130', stop_words)
    os.chdir('..')
    coll_131 = coll.parse_rcv_coll('Training131', stop_words)
    os.chdir('..')
    coll_132 = coll.parse_rcv_coll('Training132', stop_words)
    os.chdir('..')
    coll_133 = coll.parse_rcv_coll('Training133', stop_words)
    os.chdir('..')
    coll_134 = coll.parse_rcv_coll('Training134', stop_words)
    os.chdir('..')
    coll_135 = coll.parse_rcv_coll('Training135', stop_words)
    os.chdir('..')
    coll_136 = coll.parse_rcv_coll('Training136', stop_words)
    os.chdir('..')
    coll_137 = coll.parse_rcv_coll('Training137', stop_words)
    os.chdir('..')
    coll_138 = coll.parse_rcv_coll('Training138', stop_words)
    os.chdir('..')
    coll_139 = coll.parse_rcv_coll('Training139', stop_words)
    os.chdir('..')
    coll_140 = coll.parse_rcv_coll('Training140', stop_words)
    os.chdir('..')
    coll_141 = coll.parse_rcv_coll('Training141', stop_words)
    os.chdir('..')
    coll_142 = coll.parse_rcv_coll('Training142', stop_words)
    os.chdir('..')
    coll_143 = coll.parse_rcv_coll('Training143', stop_words)
    os.chdir('..')
    coll_144 = coll.parse_rcv_coll('Training144', stop_words)
    os.chdir('..')
    coll_145 = coll.parse_rcv_coll('Training145', stop_words)
    os.chdir('..')
    coll_146 = coll.parse_rcv_coll('Training146', stop_words)
    os.chdir('..')
    coll_147 = coll.parse_rcv_coll('Training147', stop_words)
    os.chdir('..')
    coll_148 = coll.parse_rcv_coll('Training148', stop_words)
    os.chdir('..')
    coll_149 = coll.parse_rcv_coll('Training149', stop_words)
    os.chdir('..')
    coll_150 = coll.parse_rcv_coll('Training150', stop_words)
    os.chdir('..')


    #q4
    bm25_weight_1 = training(coll_101, '101')
    bm25_weight_2 = training(coll_102, '102')
    bm25_weight_3 = training(coll_103, '103')
    # bm25_weight_4 = training(coll_104, '104')
    # bm25_weight_5 = training(coll_105, '105')
    bm25_weight_6 = training(coll_106, '106')
    bm25_weight_7 = training(coll_107, '107')
    bm25_weight_8 = training(coll_108, '108')
    # bm25_weight_9 = training(coll_109, '109')
    bm25_weight_10 = training(coll_110, '110')
    bm25_weight_11 = training(coll_111, '111')
    bm25_weight_12 = training(coll_112, '112')
    bm25_weight_13 = training(coll_113, '113')
    bm25_weight_14 = training(coll_114, '114')
    bm25_weight_15 = training(coll_115, '115')
    bm25_weight_16 = training(coll_116, '116')
    # bm25_weight_17 = training(coll_117, '117')
    bm25_weight_18 = training(coll_118, '118')
    # bm25_weight_19 = training(coll_119, '119')
    bm25_weight_20 = training(coll_120, '120')
    bm25_weight_21 = training(coll_121, '121')
    bm25_weight_22 = training(coll_122, '122')
    # bm25_weight_23 = training(coll_123, '123')
    # bm25_weight_24 = training(coll_124, '124')
    bm25_weight_25 = training(coll_125, '125')
    # bm25_weight_26 = training(coll_126, '126')
    bm25_weight_27 = training(coll_127, '127')
    bm25_weight_28 = training(coll_128, '128')
    bm25_weight_29 = training(coll_129, '129')
    bm25_weight_30 = training(coll_130, '130')
    # bm25_weight_31 = training(coll_131, '131')
    # bm25_weight_32 = training(coll_132, '132')
    # bm25_weight_33 = training(coll_133, '133')
    # bm25_weight_34 = training(coll_134, '134')
    # bm25_weight_35 = training(coll_135, '135')
    # bm25_weight_36 = training(coll_136, '136')
    # bm25_weight_37 = training(coll_137, '137')
    # bm25_weight_38 = training(coll_138, '138')
    # bm25_weight_39 = training(coll_139, '139')
    # bm25_weight_40 = training(coll_140, '140')
    # bm25_weight_41 = training(coll_141, '141')
    # bm25_weight_42 = training(coll_142, '142')
    # bm25_weight_43 = training(coll_143, '143')
    # bm25_weight_44 = training(coll_144, '144')
    # bm25_weight_45 = training(coll_145, '145')
    # bm25_weight_46 = training(coll_146, '146')
    # bm25_weight_47 = training(coll_147, '147')
    # bm25_weight_48 = training(coll_148, '148')
    # bm25_weight_49 = training(coll_149, '149')
    # bm25_weight_50 = training(coll_150, '150')

    # writeModel_w4(bm25_weight_1)
    writeModel_w4(bm25_weight_1, '101')
    writeModel_w4(bm25_weight_2, '102')
    writeModel_w4(bm25_weight_3, '103')
    # writeModel_w4(bm25_weight_4, '104')
    # writeModel_w4(bm25_weight_5, '105')
    writeModel_w4(bm25_weight_6, '106')
    writeModel_w4(bm25_weight_7, '107')
    writeModel_w4(bm25_weight_8, '108')
    # writeModel_w4(bm25_weight_9, '109')
    writeModel_w4(bm25_weight_10, '110')
    writeModel_w4(bm25_weight_11, '111')
    writeModel_w4(bm25_weight_12, '112')
    writeModel_w4(bm25_weight_13, '113')
    writeModel_w4(bm25_weight_14, '114')
    writeModel_w4(bm25_weight_15, '115')
    writeModel_w4(bm25_weight_16, '116')
    # writeModel_w4(bm25_weight_17, '117')
    writeModel_w4(bm25_weight_18, '118')
    # writeModel_w4(bm25_weight_19, '119')
    writeModel_w4(bm25_weight_20, '120')
    writeModel_w4(bm25_weight_21, '121')
    writeModel_w4(bm25_weight_22, '122')
    # writeModel_w4(bm25_weight_23, '123')
    # writeModel_w4(bm25_weight_24, '124')
    writeModel_w4(bm25_weight_25, '125')
    # writeModel_w4(bm25_weight_26, '126')
    writeModel_w4(bm25_weight_27, '127')
    writeModel_w4(bm25_weight_28, '128')
    writeModel_w4(bm25_weight_29, '129')
    writeModel_w4(bm25_weight_30, '130')
    # writeModel_w4(bm25_weight_31, '131')
    # writeModel_w4(bm25_weight_32, '132')
    # writeModel_w4(bm25_weight_33, '133')
    # writeModel_w4(bm25_weight_34, '134')
    # writeModel_w4(bm25_weight_35, '135')
    # writeModel_w4(bm25_weight_36, '136')
    # writeModel_w4(bm25_weight_37, '137')
    # writeModel_w4(bm25_weight_38, '138')
    # writeModel_w4(bm25_weight_39, '139')
    # writeModel_w4(bm25_weight_40, '141')
    # writeModel_w4(bm25_weight_41, '141')

    print('Current Working directory: {0}'.format(os.getcwd()))
    os.chdir('..')
    eval('101', 'dataset101-150/PTraining_benchmark_R101.txt', 'Relevance_judgments/Training101.txt')
    eval('102', 'dataset101-150/PTraining_benchmark_R102.txt', 'Relevance_judgments/Training102.txt')
    eval('103', 'dataset101-150/PTraining_benchmark_R103.txt', 'Relevance_judgments/Training103.txt')
    eval('107', 'dataset101-150/PTraining_benchmark_R107.txt', 'Relevance_judgments/Training107.txt')
    eval('108', 'dataset101-150/PTraining_benchmark_R108.txt', 'Relevance_judgments/Training108.txt')
    eval('103', 'dataset101-150/PTraining_benchmark_R103.txt', 'Relevance_judgments/Training103.txt')
    eval('103', 'dataset101-150/PTraining_benchmark_R103.txt', 'Relevance_judgments/Training103.txt')
    eval('103', 'dataset101-150/PTraining_benchmark_R103.txt', 'Relevance_judgments/Training103.txt')
    eval('103', 'dataset101-150/PTraining_benchmark_R103.txt', 'Relevance_judgments/Training103.txt')






