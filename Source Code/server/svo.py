import nltk
import numpy
import os
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.tokenize import RegexpTokenizer
from bllipparser import RerankingParser
from bllipparser import Tree
import math

'''
	jaccard similarity proper nouns
'''


class Svo():
	qsvo = ""
	svo = ""

	def word_similarity(self, word1, word2):
		temp1 = " " + word1 + " "
		temp2 = " " + word2 + " "
		if temp1 in temp2 or temp2 in temp1:
			return 1.0
		'''
		if word1 in word2 or word2 in word1:
			return 1.0
		'''
		if len([x for x in range(len(qsvo['subj'])) if qsvo['subj'][x][0]==word1 and qsvo['subj'][x][1][:3]=="NNP"]) != 0 or len([x for x in range(len(qsvo['obj'])) if qsvo['obj'][x][0]==word1 and qsvo['obj'][x][1][:3]=="NNP"]) != 0 or len([x for x in range(len(svo['subj'])) if svo['subj'][x][0]==word1 and svo['subj'][x][1][:3]=="NNP"]) != 0 or len([x for x in range(len(svo['obj'])) if svo['obj'][x][0]==word1 and svo['obj'][x][1][:3]=="NNP"]) != 0:
			if temp1 not in temp2 or temp2 not in temp1:
				return 0.0

		syn_set1 = wordnet.synsets(word1)
		syn_set2 = wordnet.synsets(word2)
		max_sim = 0
		for syn1 in syn_set1:
			for syn2 in syn_set2:
				sim = wordnet.wup_similarity(syn1,syn2)
				if max_sim < sim:
					max_sim = sim
		return max_sim

	def find_similar_word(self, word, word_set):
		max_sim = -1.0
		for ref_word in word_set:
			sim = self.word_similarity(word, ref_word)
			if sim > max_sim:
				max_sim = sim
		return max_sim

	def semantic_vector(self, words, joint_words):
		sent_set = set(words)
		semvec = numpy.zeros(len(joint_words))
		i = 0
		for joint_word in joint_words:
			semvec[i] = self.find_similar_word(joint_word,sent_set)
			i = i + 1
			return semvec

	def semantic_similarity(self, words_1,words_2):
		joint_words = set(words_1).union(set(words_2))
		vec1 = self.semantic_vector(words_1, joint_words)
		vec2 = self.semantic_vector(words_2, joint_words)
		#print numpy.dot(vec1, vec2.T)/ (math.sqrt((vec1[0]**2 + vec2[0]**2)) * math.sqrt((vec1[1]**2 + vec2[1]**2)))
		return (numpy.dot(vec1, vec2.T)/ (numpy.linalg.norm(vec1) * numpy.linalg.norm(vec2)))**2


	def getSubject(self, tokens):			#gets the subject part of the sentence
		subj = []
		#tokens = remove_stop(tokens)
		consec = False
		for token in tokens:
			if token[1][:2]=="NN" or token[1]=="PRP":
				if token[1][:2]=="NN":
					if consec:
						subj[len(subj)-1][0] += " " + token[0].lower()
					else:
						consec = True
						subj.append([token[0].lower(),token[1],1])
				else:
					consec = False
					subj.append([token[0].lower(),token[1],1])
			else:
				consec = False
				subj.append([token[0].lower(),token[1],0])
		return subj

	def getObject(self, tree, label, obj):				#gets the object part of the sentence
		if label=="NP" or label=="PP":
			tag = "NN"
		else:
			tag = "JJ"

		for subtree in tree.all_subtrees():
			if subtree.is_preterminal():
				if subtree.label[:2]==tag:
					tup = [subtree.token.lower(),(subtree.tags())[0],1]
				else:
					tup = [subtree.token.lower(),(subtree.tags())[0],0]
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
						verb.append([subtree.token.lower(),(subtree.tags())[0],1])
					else:
						verb.append([subtree.token.lower(),(subtree.tags())[0],0])
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

	def get_extra(self, segment):
		return [word[0] for word in segment if word[2]==0]

	def get_trunk(self, segment):
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
		print("sokpa")
		coef = [0.40,0.30,0.30]	#tentative values; improve for testing
		side = news.split(":")
		if len(side)==2:
			news = side[1]

		query = query.encode('utf-8')
		news = news.encode('utf-8')
		tree1 = Tree(rrp.simple_parse(str(query)))
		tree2 = Tree(rrp.simple_parse(str(news)))

		global qsvo
		qsvo = self.extract_svo(tree1[0])
		global svo
		svo = self.extract_svo(tree2[0])

		act1 = self.checkActive(svo['obj'], svo['verb'])
		act2 = self.checkActive(qsvo['obj'], qsvo['verb'])
		if not act1:
			temp = svo['subj']
			svo['subj'] = svo['obj']
			svo['obj'] = temp

		if not act2:
			temp = qsvo['subj']
			qsvo['subj'] = qsvo['obj']
			qsvo['obj'] = temp

		ssim = -1
		vsim = -1
		osim = -1

		if len(qsvo['subj'])!=0 and len(svo['subj'])!=0:
			if len(self.get_trunk(qsvo['subj']))==0 and len(self.get_trunk(svo['subj']))==0:
				ssim = self.semantic_similarity(self.get_extra(qsvo['subj']) , self.get_extra(svo['subj']))
			elif len(self.get_trunk(qsvo['subj']))==0:
				ssim = self.semantic_similarity(self.get_extra(qsvo['subj']) , self.get_trunk(svo['subj']))
			elif len(self.get_trunk(svo['subj']))==0:
				ssim = self.semantic_similarity(self.get_trunk(qsvo['subj']) , self.get_extra(svo['subj']))
			else:
				ssim = self.semantic_similarity(self.get_trunk(qsvo['subj']) , self.get_trunk(svo['subj']))

		if len(self.get_trunk(qsvo['verb']))==0 and len(self.get_trunk(svo['verb']))==0:
			coef[1] = 0
		else:
			if len(self.get_trunk(qsvo['verb']))==0:
				vsim = self.semantic_similarity(self.get_extra(qsvo['verb']) , self.get_trunk(svo['verb']))
			elif len(self.get_trunk(svo['verb']))==0:
				vsim = self.semantic_similarity(self.get_trunk(qsvo['verb']) , self.get_extra(svo['verb']))
			else:
				vsim = self.semantic_similarity(self.get_trunk(qsvo['verb']) , self.get_trunk(svo['verb']))

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

		if len(qsvo['obj'])!=0 and len(svo['obj'])!=0:
			if len(self.get_trunk(qsvo['obj']))==0 and len(self.get_trunk(svo['obj']))==0:
				osim = self.semantic_similarity(self.get_extra(qsvo['obj']) , self.get_extra(svo['obj']))
			elif len(self.get_trunk(qsvo['obj']))==0:
				osim = self.semantic_similarity(self.get_extra(qsvo['obj']) , self.get_trunk(svo['obj']))
			elif len(self.get_trunk(svo['obj']))==0:
				osim = self.semantic_similarity(self.get_trunk(qsvo['obj']) , self.get_extra(svo['obj']))
			else:
				osim = self.semantic_similarity(self.get_trunk(qsvo['obj']) , self.get_trunk(svo['obj']))

		#Check if need----------------

		if ssim < 0.35 and ssim!=-1:	#get the set containing the query
			coef[0] = 0.0
		if vsim < 0.35 and vsim!=-1:
			coef[1]	= 0.0
		#if osim < 0.5 and osim!=-1:	#object doesn't need to be reduced because of jaccard similarity
		#	coef[2] = 0.10

		return ssim*coef[0] + vsim*coef[1] + osim*coef[2]
