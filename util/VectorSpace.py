import numpy as np
import nltk

from .Parser import Parser
from . import util


class VectorSpace:
    """ A algebraic model for representing text documents as vectors of identifiers. 
    A document is represented as a vector. Each dimension of the vector corresponds to a 
    separate term. If a term occurs in the document, then the value in the vector is non-zero.
    """

    #Collection of document term vectors
    documentVectors = []
    term_freqs = {}
    #Mapping of vector index to keyword
    vectorKeywordIndex=[]
    #Tidies terms
    parser=None
    idf = []
    documents = []
    document_name = []
    lang_type = ''        

    def __init__(self, document_name=[] ,documents=[], lang_type=''):
        self.document_name = document_name
        self.documents = documents
        self.documentVectors=[]
        self.term_freqs = {}
        self.lang_type = lang_type
        self.parser = Parser()
        self.idf = []

        if(len(documents)>0):
            self.build(documents) 
    
    def feedback(self, first_retrieve, query):
        doc_name = first_retrieve[0]
        queryVector = self.buildQueryVector(query)

        feedback_index = self.document_name.index(doc_name)
        feedback_doc = self.documents[feedback_index]

        token = self.parser.tokenise(feedback_doc, self.lang_type)
        
        tagged = nltk.pos_tag(token)
        filter_tagged = []

        for tag in tagged:
            list_tag = [tag[0], tag[1]]
            list_tag[1] = list_tag[1][:2]
            filter_tagged.append(list_tag)    
            
        Noun_Verb = [tag for tag in filter_tagged if (tag[1] == 'NN') or (tag[1] == 'VB')]

        Noun_Verb = [tag[0] for tag in Noun_Verb]
        
        feedback_vector = [0] * len(self.vectorKeywordIndex)
        Noun_Verb = self.parser.removeStopWords(Noun_Verb)
        
        for word in Noun_Verb:
            try:
                feedback_vector[self.vectorKeywordIndex[word]] += 1; #Use simple Term Count Model
            except:
                pass
        
        queryVector = np.array(queryVector, dtype=np.float)
        feedback_vector = np.array(feedback_vector, dtype=np.float)

        queryVector += feedback_vector * 0.5
        
        doc_vector = np.array(self.documentVectors)
        unsort_ratings = [util.cosine(queryVector, vector) for vector in doc_vector]
        
        rating_dict = {}
        for index, score in enumerate(unsort_ratings):
            rating_dict[self.document_name[index]] = score

        rating_list = []
        for key, value in rating_dict.items():
            temp = [key,value]
            rating_list.append(temp)

        rating_list = sorted(rating_list, key=lambda x: x[1], reverse=True)

        return rating_list[:10]
        

    def calc_idf(self, ):
        doc_num = len(self.documentVectors)
        self.idf = [doc_num] * len(self.term_freqs)
        #idf must be log
        for i, word in enumerate(self.term_freqs):
            self.idf[i] /= self.term_freqs[word]
        self.idf = np.log10(self.idf)

    def build(self, documents):
        """ Create the vector space for the passed document strings """
        self.vectorKeywordIndex = self.getVectorKeywordIndex(documents)
        self.documentVectors = [self.makeVector(document, 'doc') for document in documents]
        self.calc_idf()      

    def getVectorKeywordIndex(self, documentList):
        """ create the keyword associated to the position of the elements within the document vectors """

        #Mapped documents into a single word string	
        vocabularyString = " ".join(documentList)

        vocabularyList = self.parser.tokenise(vocabularyString, self.lang_type)
        #Remove common words which have no search value
        vocabularyList = self.parser.removeStopWords(vocabularyList)
        uniqueVocabularyList = util.removeDuplicates(vocabularyList)

        vectorIndex={}
        offset=0
        #Associate a position with the keywords which maps to the dimension on the vector used to represent this word
        for word in uniqueVocabularyList:
            self.term_freqs[word] = 0
            vectorIndex[word]=offset
            offset+=1
        return vectorIndex  #(keyword:position)


    def makeVector(self, wordString, doc_type=''):
        """ @pre: unique(vectorIndex) """
        vector = [0] * len(self.vectorKeywordIndex)
        #Initialise vector with 0's
        wordList = self.parser.tokenise(wordString, self.lang_type)
        wordList = self.parser.removeStopWords(wordList)

        if (doc_type == 'doc'):
            tokenize_doc = util.removeDuplicates(wordList)
            for word in tokenize_doc:
                # doc freq of t
                self.term_freqs[word] += 1

        for word in wordList:
            try:
                vector[self.vectorKeywordIndex[word]] += 1; #Use simple Term Count Model
            except:
                pass
        return vector


    def buildQueryVector(self, termList):
        """ convert query string into a term vector """
        ## termList is a string
        query = self.makeVector(" ".join(termList))
        return query


    def related(self,documentId):
        """ find documents that are related to the document indexed by passed Id within the document Vectors"""
        ratings = [util.cosine(self.documentVectors[documentId], documentVector) for documentVector in self.documentVectors]
        return ratings


    def search(self,searchList, document_name, weight_scheme, similarity):
        """ search for documents that match based on a list of terms """
        queryVector = self.buildQueryVector(searchList)

        unsort_ratings = []
        
        if weight_scheme == 'tf':
            if(similarity == 'Cosine Similarity'):
                unsort_ratings = [util.cosine(queryVector, documentVector) for documentVector in self.documentVectors]

            elif(similarity == 'Euclidean Distance'): 
                unsort_ratings = [util.euclidean(queryVector, documentVector) for documentVector in self.documentVectors]

        elif weight_scheme == 'tfidf':
            doc_vector = np.array(self.documentVectors)
            query_vector = np.array(queryVector)

            doc_vector = np.multiply(doc_vector, self.idf)
            query_vector = np.multiply(query_vector, self.idf)

            if(similarity == 'Cosine Similarity'):
                unsort_ratings = [util.cosine(query_vector, vector) for vector in doc_vector]

            elif(similarity == 'Euclidean Distance'): 
                unsort_ratings = [util.euclidean(query_vector, vector) for vector in doc_vector]

        rating_dict = {}
        for index, score in enumerate(unsort_ratings):
            rating_dict[document_name[index]] = score

        rating_list = []
        for key, value in rating_dict.items():
            temp = [key,value]
            rating_list.append(temp)

        if(similarity == 'Cosine Similarity'):
            rating_list = sorted(rating_list, key=lambda x: x[1], reverse=True)
        elif (similarity == 'Euclidean Distance'):
            rating_list = sorted(rating_list, key=lambda x: x[1])

        return rating_list[:10]
