import os
import argparse
from util.VectorSpace import VectorSpace

parser = argparse.ArgumentParser()
parser.add_argument("--queryEn",
                    type=str,
                    help="Input English text query")
parser.add_argument("--queryCh",
                    type=str,
                    help="Input Chinese text query")

args = parser.parse_args()

def question_one_two():
    print("Question 1")
    print("------------------")

    path = './EnglishNews/'
    # doc name list
    document_name = os.listdir(path)
    documents = []

    # about 10 seconds, not bottleneck
    for name in document_name:
        if os.path.isfile(path + name):
            with open(path + name, 'r', encoding="utf-8") as f:
                str = f.read()
                # the order is same as document_nameList
                documents.append(str)

    vector_space = VectorSpace(document_name ,documents)

    similarity = ['Cosine Similarity', 'Euclidean Distance']
    feedback_doc = []

    for matrix in similarity:
        rating_list = vector_space.search([args.queryEn], document_name, 'tfidf', matrix)
    
        print(f"TF-IDF Weighting + {matrix}:")
        print('NewsID\t\tscore')

        for i, doc in enumerate(rating_list):
            print(f'{doc[0]}\t{doc[1]}')

        print('\n')
    
        if (matrix == 'Cosine Similarity'):
            feedback_doc = rating_list[0]
    
    print("Question 2")
    print("------------------")

    feedback_list = vector_space.feedback(feedback_doc ,[args.queryEn])

    print(f"TF-IDF Weighting + Cosine Similarity + relevance feedback")
    print('NewsID\t\tscore')
    for i, doc in enumerate(feedback_list):
        print(f'{doc[0]}\t{doc[1]}')

    print('\n')

def question_three():
    print("Question 3")
    print("------------------")

    path = './ChineseNews/'
    # doc name list
    document_name = os.listdir(path)
    documents = []

    # about 10 seconds, not bottleneck
    for name in document_name:
        if os.path.isfile(path + name):
            with open(path + name, 'r', encoding="utf-8") as f:
                str = f.read()
                # the order is same as document_nameList
                documents.append(str)
    
    vector_space = VectorSpace(document_name ,documents, 'chinese')

    weight_scheme = ['tf', 'tfidf']
    similarity = 'Cosine Similarity'
    
    for scheme in weight_scheme:

        rating_list = vector_space.search([args.queryCh], document_name, scheme, similarity)

        print(f"{scheme.upper()} Weighting + {similarity}")
        print('NewsID\t\tscore')

        for i, doc in enumerate(rating_list):
            print(f'{doc[0]}\t{doc[1]}')

        print('\n')

    return

def question_four():
    print("Question 4")
    print("------------------")

    path = './smaller_dataset/collections/'
    # doc name list
    document_name = os.listdir(path)
    documents = []

    # about 10 seconds, not bottleneck
    for name in document_name:
        if os.path.isfile(path + name):
            with open(path + name, 'r', encoding="utf-8") as f:
                str = f.read()
                # the order is same as document_nameList
                documents.append(str)
    
    path = './smaller_dataset/queries/'
    # doc name list
    query_name = os.listdir(path)
    queris = []

    # about 10 seconds, not bottleneck
    for name in query_name:
        if os.path.isfile(path + name):
            with open(path + name, 'r', encoding="utf-8") as f:
                str = f.read()
                # the order is same as document_nameList
                queris.append(str)

    path = './smaller_dataset/rel.tsv'
    rel_tsv = {}
    with open(path, 'r', encoding="utf-8") as f:
        while True:
            line = f.readline()
            if not line: break
            line = line.strip().split("\t")
            line[1] = line[1].replace('[', "")
            line[1] = line[1].replace(']', "")
            line[1] = line[1].replace(',', "")
            line[1] = line[1].split(' ')
            rel_tsv[line[0]] = line[1]

    vector_space = VectorSpace(document_name ,documents)
    
    matrix = 'Cosine Similarity'
    MAP_list = []
    MRR_list = []
    Recall_list = []

    for query_index, query in enumerate(queris):
        rating_list = vector_space.search([query], document_name, 'tfidf', matrix)
        ## queris in the corpus are not well ordered. 
        query_ID = query_name[query_index].replace('.txt', '')

        ### [name, relevent(0/1), recall, precision]
        predict_doc = []
        for k, doc in enumerate(rating_list):
            item = doc[0].replace('d', '').replace('.txt', '')
            predict_doc.append([item, 0, 0, 0])

        for doc in predict_doc:
            if(doc[0] in rel_tsv[query_ID]):
                doc[1] = 1

        rel_num = len(rel_tsv[query_ID])
        rel_retrived = 0
        precision_sum = 0
        MRR_rank = 0
        find_firstRel = False

        for j, doc in enumerate(predict_doc):
            rel_retrived += doc[1]
            if (rel_retrived == 1) and (find_firstRel == False):
                MRR_rank = 1/(j+1)
                find_firstRel = True
            doc[2] = rel_retrived / rel_num
            doc[3] = rel_retrived / (j+1)
            if doc[1] == 1:
                precision_sum += doc[3]

        if rel_retrived != 0:
            average_precision = precision_sum / rel_retrived
        else: average_precision = 0

        MAP_list.append(average_precision)
        MRR_list.append(MRR_rank)

        #recall
        tp = rel_retrived
        fn = rel_num - rel_retrived

        recall = tp / (tp + fn)
        Recall_list.append(recall)

    len_queris = len(queris)

    MAP = sum(MAP_list) / len_queris
    MRR = sum(MRR_list) / len_queris
    RECALL = sum(Recall_list) / len_queris

    print("tfidf retrive...")
    print("------------------")
    
    print(f'tridf\tMRR@10\t{MRR}')
    print(f'tridf\tMAP@10\t{MAP}')
    print(f'tridf\tRECALL@10\t{RECALL}')

    print("------------------")
    return

if __name__ == '__main__':
    question_one_two()
    question_three()
    question_four()
    
    
    