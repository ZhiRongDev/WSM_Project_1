#http://tartarus.org/~martin/PorterStemmer/python.txt
from .PorterStemmer import PorterStemmer
import jieba

class Parser:

	#A processor for removing the commoner morphological and inflexional endings from words in English
	stemmer=None

	stopwords=[]

	def __init__(self,):
		self.stemmer = PorterStemmer()

		#English stopwords from ftp://ftp.cs.cornell.edu/pub/smart/english.stop
		english_stopwords = open('english.stop', 'r', encoding="utf-8").read().split()
		chinese_stopwords = open('chinese.stop', 'r', encoding="utf-8").read().split()
		
		self.stopwords = english_stopwords + chinese_stopwords	

	def clean(self, string):
		""" remove any nasty grammar tokens from string """
		string = string.replace(".","")
		string = string.replace("\n","")
		string = string.replace(",","")
		string = string.replace(":","")
		string = string.replace(":","")
		string = string.replace("[","")
		string = string.replace("]","")
		string = string.replace(")","")
		string = string.replace("(","")
		string = string.replace("{","")
		string = string.replace("}","")
		string = string.replace("/","")
		string = string.replace("?","")
		string = string.replace("!","")
		string = string.replace("’","")
		string = string.replace("“","")
		string = string.replace("”","")
		string = string.replace("'","")
		string = string.replace('"',"")
		string = string.replace('”',"")
		string = string.replace(';',"")
		string = string.replace('{',"")
		string = string.replace('}',"")
		string = string.replace('。',"")
		string = string.replace('!',"")
		string = string.replace("\s+"," ")
		string = string.lower()
		return string
	

	def removeStopWords(self,list):
		""" Remove common words which have no search value """
		return [word for word in list if word not in self.stopwords ]


	def tokenise(self, string, lang_type=''):
		""" break string up into tokens and stem words """
		string = self.clean(string)
		
		# jieba
		if(lang_type == 'chinese'):
			string = " ".join(jieba.cut(string))
			
		words = string.split(" ")

		return [self.stemmer.stem(word,0,len(word)-1) for word in words]