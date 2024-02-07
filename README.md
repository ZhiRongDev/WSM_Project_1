# WSM Project1: Ranking by Vector Space Models
###### WSM Project


## package dependencies
Please install the packages below.

1. **jieba**
https://github.com/fxsjy/jieba

2. **NLTK**
https://www.nltk.org/


### Command Format
```
$ python main.py --queryEn <English query> --queryCh <Chinese query>
```

#### Example
```
$ python main.py --queryEn "Youtube Taiwan COVID-19" --queryCh "烏克蘭 大選"
```
### folder structure
```
.
├── main.py
├── chinese.stop
├── english.stop
├── README.md
├── .gitignore
├── Project1_intro.pdf
├── util/
    ├── __init__.py
    ├── Parser.py
    ├── PostStemmer.py
    ├── util.py
    └── VectorSpace.py
├── ChineseNews/
├── EnglishNews/
└── smaller_dataset/
    ├── collections/
    ├── queries/
    └── rel.tsv
```

### Ouput
```
Question 1
------------------
TF-IDF Weighting + Cosine Similarity:
NewsID          score
News1240.txt    0.49006005986044865
News7403.txt    0.4243846846544435
News1679.txt    0.42189966264049933
News2230.txt    0.4017744517162297
News668.txt     0.35115792064754686
News623.txt     0.3459954097258833
News7362.txt    0.31448137803852155
News796.txt     0.2827555613612497
News2401.txt    0.2760272141647473
News7570.txt    0.2760272141647473


TF-IDF Weighting + Euclidean Distance:
NewsID          score
News2925.txt    5.199594659555012
News1830.txt    5.2541801396867935
News2424.txt    5.311503021974884
News7207.txt    5.311503021974884
News1994.txt    5.3887195306754005
News7467.txt    5.516592224585441
News7098.txt    5.626412349234616
News2401.txt    5.771357789506635
News7570.txt    5.771357789506635
News1497.txt    5.828576775208701


Question 2
------------------
TF-IDF Weighting + Cosine Similarity + relevance feedback
NewsID          score
News1240.txt    0.8541666666666667
News2230.txt    0.3638034375544995
News7403.txt    0.329914439536929
News1679.txt    0.32659863237109044
News623.txt     0.3263956049169334
News668.txt     0.3221916685210339
News1198.txt    0.2672612419124244
News820.txt     0.2608745973749755
News3857.txt    0.2528083175591482
News796.txt     0.242535625036333


Question 3
------------------
Building prefix dict from the default dictionary ...
Loading model from cache C:\Users\user\AppData\Local\Temp\jieba.cache
Loading model cost 0.738 seconds.
Prefix dict has been built successfully.
TF Weighting + Cosine Similarity
NewsID          score
News200049.txt  0.8442943641657992
News200053.txt  0.8321537350211664
News200159.txt  0.8259695049376905
News200892.txt  0.8256230889574704
News200125.txt  0.8243747455768934
News200047.txt  0.8237243101489933
News200156.txt  0.8235674745742362
News200004.txt  0.8228834755496903
News200613.txt  0.8204452822390552
News200140.txt  0.8202033079357383


TFIDF Weighting + Cosine Similarity
NewsID          score
News200049.txt  0.2093471862623416
News200892.txt  0.20029138540887295
News200053.txt  0.18300493448699665
News200004.txt  0.1786810716188895
News200071.txt  0.17756357258792474
News200847.txt  0.17675097321718916
News200156.txt  0.16862197115632946
News200056.txt  0.16556677658661528
News200908.txt  0.16233991255374366
News200137.txt  0.16170965815854282


Question 4
------------------
tfidf retrive...
------------------
tridf   MRR@10  0.6220916875522139
tridf   MAP@10  0.531516197487767
tridf   RECALL@10       0.14643432444630522

```# WSM_Project_1
