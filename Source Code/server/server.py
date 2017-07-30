'''
Fact U App - A mobile app for news verification using hidden markov model and sentence similary analysis

Emmanuel B. Constantino Jr.
2013-08147

Homer C. Malijan
2013-09022
'''
import socket
import sqlite3
import sys
import nltk
import random
import socket
import math
import nltk
import numpy
import os
import datetime
import newspaper
import re

from urllib.parse import urlparse
from newspaper import Article
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.tokenize import RegexpTokenizer
from bllipparser import RerankingParser
from bllipparser import Tree

# coding: utf-8

# connect to database
conn = sqlite3.connect('db/newsDb.db')
cur = conn.cursor()
print("Succesfully connected to DB!")

# load news model for sentence similarity analysis
global rrp
rrp = RerankingParser.fetch_and_load('WSJ+Gigaword-v2')
print("Succesfully created model!")

class HMM():
    def set_up(self, tup):
        #sorts the data according to date
        sort = sorted([c[1] for c in tup], key=lambda x: datetime.datetime.strptime(x, '%Y-%m-%d'))

        for i in range(len(sort)):
        	for c in tup:
        		if sort[i]==c[1]:
        			sort[i] = c

        temp1 = {}
        temp2 = {}

        tup = sort

        #set up of initial value of parameters for the HMM
        for i in range(len(tup)):
                #parsing of news string
        	arr = tup[i][0].split("***")
        	string = "news"+str(i+1)
                #arr[3] == 1 means that the news is verified; temp1 gets the score while temp2 gets the complement
        	if arr[3]=='1':
        		temp1[string] = float(arr[1])
        		temp2[string] = 1.0-float(arr[1])
                #this means that the news is satiric; temp1 gets the complement while temp2 gets the score
        	else:
        		temp1[string] = 1.0-float(arr[1])
        		temp2[string] = float(arr[1])
        #set up of emission probability
        emit_p = {}
        emit_p['Verified'] = temp1
        emit_p['Satiric'] = temp2

        #set up of observations
        obs = ()
        for i in range(len(tup)):
        	string = "news"+str(i+1)
        	obs = obs + (string,)

        states = ('Verified', 'Satiric')
        #initial values of start probability and transition probability
        start_p = {'Verified': 0.5, 'Satiric': 0.5}
        trans_p = {
           'Verified' : {'Verified': 0.7, 'Satiric': 0.3},
           'Satiric' : {'Verified': 0.3, 'Satiric': 0.7}
           }
        status = self.viterbi(obs, states, start_p, trans_p, emit_p)
        return(status)

    #code snippet lifted from https://en.wikipedia.org/wiki/Viterbi_algorithm
    def viterbi(self, obs, states, start_p, trans_p, emit_p):
         V = [{}]
         for st in states:
             V[0][st] = {"prob": start_p[st] * emit_p[st][obs[0]], "prev": None}
         # Run Viterbi when t > 0
         for t in range(1, len(obs)):
             V.append({})
             for st in states:
                 max_tr_prob = max(V[t-1][prev_st]["prob"]*trans_p[prev_st][st] for prev_st in states)
                 for prev_st in states:
                     if V[t-1][prev_st]["prob"] * trans_p[prev_st][st] == max_tr_prob:
                         max_prob = max_tr_prob * emit_p[st][obs[t]]
                         V[t][st] = {"prob": max_prob, "prev": prev_st}
                         break
         opt = []
         # The highest probability
         max_prob = max(value["prob"] for value in V[-1].values())
         previous = None
         # Get most probable state and its backtrack
         for st, data in V[-1].items():
             if data["prob"] == max_prob:
                 opt.append(st)
                 previous = st
                 break
         # Follow the backtrack till the first observation
         for t in range(len(V) - 2, -1, -1):
             opt.insert(0, V[t + 1][previous]["prev"])
             previous = V[t + 1][previous]["prev"]

         return opt[len(opt)-1]

class SSA():
    qsvo = ""
    svo = ""
    def word_similarity(self, word1, word2):
        temp1 = " " + word1 + " "
        temp2 = " " + word2 + " "
        #returns 1.0 if the words are equal or is a substring
        if temp1 in temp2 or temp2 in temp1:
                return 1.0
        #checks if the either of the words is a Proper noun, meaning that the word should be present to each other
        if len([x for x in range(len(qsvo['subj'])) if qsvo['subj'][x][0]==word1 and qsvo['subj'][x][1][:3]=="NNP"]) != 0 or len([x for x in range(len(qsvo['obj'])) if qsvo['obj'][x][0]==word1 and qsvo['obj'][x][1][:3]=="NNP"]) != 0 or len([x for x in range(len(svo['subj'])) if svo['subj'][x][0]==word1 and svo['subj'][x][1][:3]=="NNP"]) != 0 or len([x for x in range(len(svo['obj'])) if svo['obj'][x][0]==word1 and svo['obj'][x][1][:3]=="NNP"]) != 0:
            if temp1 not in temp2 or temp2 not in temp1:
		              return 0.0

        #gets the synonym sets of each word
        syn_set1 = wordnet.synsets(word1)
        syn_set2 = wordnet.synsets(word2)
        max_sim = 0
        #compares each synonym with the use of wup_similarity and gets the highest score
        for syn1 in syn_set1:
            for syn2 in syn_set2:
                sim = wordnet.wup_similarity(syn1,syn2)
                if (sim == None):
                    sim = 0
                if (max_sim < sim):
                    max_sim = sim
        return max_sim

    def find_similar_word(self, word, word_set):	#finds the most similar word in the other sentence
    	max_sim = -1.0
    	for ref_word in word_set:
    		sim = self.word_similarity(word, ref_word)
    		if sim > max_sim:
    			max_sim = sim
    	return max_sim

    def semantic_vector(self, words, joint_words):
        sent_set = set(words)
        semvec = numpy.zeros(len(joint_words))	#creates a set of zeroes
        i = 0
        for joint_word in joint_words:
            semvec[i] = self.find_similar_word(joint_word,sent_set)	#gets the similarity score for each word in the joint set
            i = i + 1
        return semvec

    def semantic_similarity(self, words_1,words_2):
        joint_words = set(words_1).union(set(words_2))
        vec1 = self.semantic_vector(words_1, joint_words)
        vec2 = self.semantic_vector(words_2, joint_words)
        return (numpy.dot(vec1, vec2.T)/ (numpy.linalg.norm(vec1) * numpy.linalg.norm(vec2)))**2	#cosine similarity formula

    def getSubject(self, tokens):			#gets the subject part of the sentence
        subj = []
        consec = False
        for token in tokens:
            if token[1][:2]=="NN" or token[1]=="PRP":
                if token[1][:2]=="NN":
                    if consec:
                        subj[len(subj)-1][0] += " " + token[0]
                    else:
                        consec = True
                        subj.append([token[0],token[1],1])
                else:
                    consec = False
                    subj.append([token[0],token[1],1])
            else:
                consec = False
                subj.append([token[0],token[1],0])
        return subj

    def getObject(self, tree, label, obj):				#gets the object part of the sentence
        if label=="NP" or label=="PP":
            tag = "NN"
        else:
            tag = "JJ"

        for subtree in tree.all_subtrees():
            if subtree.is_preterminal():
                if subtree.label[:2]==tag:
                    tup = [subtree.token,(subtree.tags())[0],1]
                else:
                    tup = [subtree.token,(subtree.tags())[0],0]
                if tup not in obj:
                    obj.append(tup)

    def remove_stop(self, word_list):			#removes stop words in the sentence
    	filtered_words = [word for word in word_list if word[0].lower() not in stopwords.words('english')]
    	return filtered_words

    def extract_svo(self, tree):
        verb = []
        obj = []
        objPart = False
        other = []
        for subtree in tree:
            if subtree.label=="NP" or subtree.label=="VP":
                if subtree.label=="NP":							#getting the subject
                    subj=self.getSubject(subtree.tokens_and_tags())
                if subtree.label=="VP":
                    vp = subtree
                    break
            else:
                other.append(subtree.tokens()) 						#getting other parts
        for subtree in vp.all_subtrees():
            if not objPart:
                if subtree.is_preterminal():						#getting the verb
                    if subtree.label[:2]=="VB" and subtree.token.lower() not in stopwords.words('english'):
                        extracted_verb = subtree.token.lower()
                        verb.append([extracted_verb,(subtree.tags())[0],1])
                    else:
                        verb.append([subtree.token,(subtree.tags())[0],0])
                if subtree.label=="NP" or subtree.label=="PP" or subtree.label=="ADJP":
                    objPart=True
            if objPart:									#getting the object
                if subtree.label=="NP" or subtree.label=="PP" or subtree.label=="ADJP":
                    self.getObject(subtree,subtree.label,obj)
        return {'subj' : subj, 'verb' : verb, 'obj' : obj, 'other' : other}

    '''
    	Advantage:
    		- More precise since segmented part are really compared.
    	Disadvantage:
    		- Largely depends on the parser's evaluation
    		- News don't really always have a sentence for news title
    '''

    def get_extra(self, segment):			#gets parts of the segment that are only modifiers or non-significant words
    	return [word[0] for word in segment if word[2]==0]

    def get_trunk(self, segment):			#gets the main parts of the segment
    	return [word[0] for word in segment if word[2]==1]

    def checkActive(self, obj, verb):			#determines if sentence is in passive form
        counter = 0
        for v in verb:
            if v[1][:2]=="VB":
                counter+=1

        if counter==1:
            return True

        for o in obj:
            if o[0]=="by":
                return False
        ha = False
        be = False
        vbn = False
        for v in verb:
            if v[0]=="has" or v[0]=="have" or v[0]=="had" or v[0]=="having":
                ha = True
            if v[0]=="be" or v[0]=="being" or v[0]=="been":
                be = True
            if v[1]=="VBN":
                vbn = True
        if vbn:
            if be:
                return False
            if ha:
                return True
            return False

        return True

    def grammarApproach(self, query, news, rrp):		#Grammar structure approach

        coef = [0.40,0.30,0.30]	#tentative values; improve for testing

        side = news.split(":")
        if len(side)==2:
    	    news = side[1]

        tree1 = Tree(rrp.simple_parse(query))
        tree2 = Tree(rrp.simple_parse(news))

        global qsvo
        qsvo = self.extract_svo(tree1[0])

        global svo
        svo = self.extract_svo(tree2[0])

        act1 = self.checkActive(svo['obj'], svo['verb'])
        act2 = self.checkActive(qsvo['obj'], qsvo['verb'])
        if not act1:			#the first sentence is in passive form, subject and object are reversed
    	    temp = svo['subj']
    	    svo['subj'] = svo['obj']
    	    svo['obj'] = temp

        if not act2:			#the second sentence is in passive form, subject and object are reversed
    	    temp = qsvo['subj']
    	    qsvo['subj'] = qsvo['obj']
    	    qsvo['obj'] = temp

        ssim = -1
        vsim = -1
        osim = -1
        #gets the similarity score of the subjects
        if len(qsvo['subj'])!=0 and len(svo['subj'])!=0:
            if len(self.get_trunk(qsvo['subj']))==0 and len(self.get_trunk(svo['subj']))==0:
                ssim = self.semantic_similarity(self.get_extra(qsvo['subj']) , self.get_extra(svo['subj']))
            elif len(self.get_trunk(qsvo['subj']))==0:
                ssim = self.semantic_similarity(self.get_extra(qsvo['subj']) , self.get_trunk(svo['subj']))
            elif len(self.get_trunk(svo['subj']))==0:
                ssim = self.semantic_similarity(self.get_trunk(qsvo['subj']) , self.get_extra(svo['subj']))
            else:
    	        ssim = self.semantic_similarity(self.get_trunk(qsvo['subj']) , self.get_trunk(svo['subj']))
        #does not consider the verb when both verbs are linking verbs
        if len(self.get_trunk(qsvo['verb']))==0 and len(self.get_trunk(svo['verb']))==0:
    	    coef[1] = 0
        #gets the similarity score of the verbs
        else:
    	    if len(self.get_trunk(qsvo['verb']))==0:
    		    vsim = self.semantic_similarity(self.get_extra(qsvo['verb']) , self.get_trunk(svo['verb']))
    	    elif len(self.get_trunk(svo['verb']))==0:
    		    vsim = self.semantic_similarity(self.get_trunk(qsvo['verb']) , self.get_extra(svo['verb']))
    	    else:
    		    vsim = self.semantic_similarity(self.get_trunk(qsvo['verb']) , self.get_trunk(svo['verb']))
        #set up of weighing coefficients
        if not act1 or not act2:
    	    if ssim==-1:
    		    if coef[1] == 0:
    			    coef = [0.0, 0.0, 1.0]
    		    else:
    			    coef = [0.0, 0.5, 0.5]
    	    elif len(qsvo['obj'])==0:
    		    if coef[1] == 0:
    			    coef = [1.0, 0.0, 0.0]
    		    else:
    			    coef = [0.5,0.5,0.0]
        else:
    	    if len(qsvo['obj'])==0:
    		    if coef[1] == 0:
    			    coef = [1.0, 0.0, 0.0]
    		    else:
    			    coef = [0.5,0.5,0.0]
    	    else:
    		    if coef[1] == 0:
    		        coef = [0.7, 0.0, 0.3]
        #gets the similarity score of objects
        if len(qsvo['obj'])!=0 and len(svo['obj'])!=0:
            if len(self.get_trunk(qsvo['obj']))==0 and len(self.get_trunk(svo['obj']))==0:
                osim = self.semantic_similarity(self.get_extra(qsvo['obj']) , self.get_extra(svo['obj']))
            elif len(self.get_trunk(qsvo['obj']))==0:
                osim = self.semantic_similarity(self.get_extra(qsvo['obj']) , self.get_trunk(svo['obj']))
            elif len(self.get_trunk(svo['obj']))==0:
                osim = self.semantic_similarity(self.get_trunk(qsvo['obj']) , self.get_extra(svo['obj']))
            else:
    	        osim = self.semantic_similarity(self.get_trunk(qsvo['obj']) , self.get_trunk(svo['obj']))

        if ssim < 0.35 and ssim!=-1:
    	    coef[0] = 0.0
        if vsim < 0.35 and vsim!=-1:
    	    coef[1]	= 0.0
        return ssim*coef[0] + vsim*coef[1] + osim*coef[2]

    '''
    Parse every output from the database and compute similarity with the input Statement
    take statements whose score is greather than or equal to 0.749
    '''

    def ssa(self, query, news):
        temparr = []
        resultarr = []
        tempMaxV = 0
        tempMaxS = 0
        newsBuscuit = 'NewsBiscuit'
        query = query.replace('‘','')
        query = query.replace('’','')

        # parse every statement returned from the database
        for i in range(len(news)):
            temp = news[i].split("***")
            try:
                # remove NewsBiscuit if it exist in the output from thedatabase
                temp[0] = temp[0].replace('NewsBiscuit','')
                # remove special character from pdf's
                temp[0] = temp[0].replace('‘','')
                temp[0] = temp[0].replace('’','')

                # compute score
                score = self.grammarApproach(str(query), str(temp[0]), rrp)
                print(query,temp[0],score)

                # store information in an array if score passes 0.749 threshold
                if score >= 0.749:
                    if temp[3] == '1' and score > tempMaxV:
                        tempMaxV = score
                    elif temp[3] == '0' and score > tempMaxS:
                        tempMaxS = score
                    tempString = temp[0] + "***" + str(score) + "***" + temp[2] + "***" + temp[3]
                    print(tempString)
                    temparr.append((tempString, temp[1]))
                    resultarr.append((temp[0] + "***" + temp[2] + "||" ,score))
            except:
                pass

        # if array for HMM computation is empty, return insufficient evidence
        if(not temparr):
            return(("Insufficient evidence", ""))

        # load retrieved data to HMM model for classification
        hmm = HMM()
        result = hmm.set_up(temparr)
        temp_resultarr = sorted(resultarr, key=lambda x: x[1])

        finalArr = ''
        counter1 = 0
        for t in temp_resultarr:
            if counter1==2:
                break
            finalArr = t[0] + finalArr
            counter1 = counter1 + 1
        print(finalArr)

        # return appropriate results
        if 'Verified' in result:
            return((str(str(tempMaxV*100)+ "% " + result), str(finalArr)))
        else:
            return((str(str(tempMaxS*100)+ "% " + result), str(finalArr)))

'''
convert input string into an SQL query
'''
def processInput(input):
    try:
        # try to take subject from input
        input_tree = Tree(rrp.simple_parse(input))
        temp = SSA()
        x = temp.extract_svo(input_tree[0])
        print(x['subj'])
        toParse = input.split(" ")

        # create a query that will take all the rows in the database that contains
        # the subject of the input
        sql = "SELECT TITLE, PUBLISH_DATE, URL, LEGIT FROM news WHERE "
        sql += 'TITLE LIKE "%'
        sql += str(x['subj'][0][0])
        sql += '%"'

        return sql
    except:
        # if an error occured while trying to take the subect
        print("HELLO WORLD")
        toParse = input.split(" ")

        # create a query that will take all the rows in the database that
        # contains the words in the input skipping all the stopwords
        sql = "SELECT TITLE, PUBLISH_DATE, URL, LEGIT FROM news WHERE "
        for temp in toParse:
            if(temp.lower() in stopwords.words('english')):
                if (temp == toParse[-1]):
                    break
                continue
            sql += 'TITLE LIKE "%'
            sql += temp
            sql += '%"'
            if (temp == toParse[-1]):
                break
            sql += ' OR '
        return sql

'''
remove uneccesary characters caused by sending data
'''
def processData(input):
    l = list(str(input))
    del(l[1])
    del(l[0])
    del(l[len(input)])
    data = "".join(l)  # convert back to string

    return(data)

def Main():
    host = '0.0.0.0'
    port = 8081

    #bind
    s = socket.socket()
    s.bind((host, port))

    s.listen(1)

    #continuous connection
    while True:
        #wait for request
        c,addr = s.accept()
        print ("Connection from: " + str(addr))

        #receive data
        data = c.recv(1024)
        if not data:
            print("No Data Recieved")
            continue
        else:
            input = data.decode()
            print(input)

        score = SSA()
        #preprocess string before query

        data = processData(data)
        o = urlparse(data)
        if(o.scheme!='' and o.netloc!=''):
            try:
                a = Article(data)
                a.download()
                a.parse()
                input = a.title
                print(input)
            except:
                c.send(str.encode(('Cannot access link', '')
))
                c.close
                continue
        else:
            print("==========================NOT A URL")

        sql = processInput(input)
        #Execute select
        print(sql)
        cur.execute(sql)


        news = []
        #send row 1 by 1
        for row in cur:
            a = "".join(str(row[0]))
            b = "".join(str(row[1]))
            d = "".join(str(row[2]))
            e = "".join(str(row[3]))
            toSend = a + "***" + b.split(' ')[0] + "***" + d + "***" + e + ""

            news.append(toSend)
            #c.send(str.encode(a))
        #send end message
        result = score.ssa(data, news)
        print(result)
        c.send(str.encode(str(result)))
        print("sent!")
        #close connection
        c.close

    #close database connection
    conn.close()

if __name__ == "__main__":
        Main()
