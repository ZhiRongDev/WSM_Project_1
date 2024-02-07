import sys
import numpy as np

#http://www.scipy.org/
try:
	from numpy import dot
	from numpy.linalg import norm
except:
	print("Error: Requires numpy from http://www.scipy.org/. Have you installed scipy?")
	sys.exit() 

def removeDuplicates(list):
	""" remove duplicates from a list """
	return set((item for item in list))

def cosine(vector1, vector2):
	""" related documents j and q are in the concept space by comparing the vectors :
		cosine  = ( V1 * V2 ) / ||V1|| x ||V2|| """
	return float(np.dot(vector1,vector2) / (norm(vector1) * norm(vector2)))

# the smaller the better
def euclidean(vector1, vector2):
	vector1 = np.array(vector1)
	vector2 = np.array(vector2)
	return float(norm(vector1-vector2))