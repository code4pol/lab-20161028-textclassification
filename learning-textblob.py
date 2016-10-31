#
# Revisao
#
from textblob.classifiers import NaiveBayesClassifier

treinamento = [ ('Eu sou lindo', 'A'),
				('Eu sou feio','B')]

c = NaiveBayesClassifier(treinamento)

print(c.classify('seu feio')) 		# B
print(c.classify('seu horroroso'))	# B
print(c.classify('seu bonito'))		# B
print(c.classify('seu lindo'))		# A

print('labels:',c.labels())			# labels: ['A', 'B']

test = [ ('Voce e muito gato','A'),
		 ('Voce e muito feio','B')]

print('acuracia:',c.accuracy(test))	# acuracia: 0.5

test = [ ('Voce e muito lindo','A'),
		 ('Voce e muito feio','B')]

print('acuracia:',c.accuracy(test))	# acuracia: 1.0

print('features:',c.extract_features('Eu sou horroroso'))
# features: {'contains(Eu)': True, 'contains(sou)': True, 'contains(lindo)': False, 'contains(feio)': False}

c.show_informative_features()
# Most Informative Features
#            contains(sou) = True                B : A      =      1.0 : 1.0
#             contains(Eu) = True                B : A      =      1.0 : 1.0

#
# So que o pacote textblob eh mais do que 
# classificacao de texto. Vejamos.
#
from textblob import TextBlob
text = TextBlob("I went home. Because I'm happy. Clap along if you feel like a room without a roof.")

print('text:',text)
# text: I went home. Because I'm happy. Clap along if you feel like a room without a roof.

print('sentences:',text.sentences)
# sentences: [Sentence("I went home."), Sentence("Because I'm happy."), Sentence("Clap along if you feel like a room without a roof.")]

print('words:',text.words)
# words: ['I', 'went', 'home', 'Because', 'I', "'m", 'happy', 'Clap', 'along', 'if', 'you', 'feel', 'like', 'a', 'room', 'without', 'a', 'roof']

print('tags:',text.tags)
# tags: [('I', 'PRP'), ('went', 'VBD'), ('home', 'NN'), ('Because', 'IN'), ('I', 'PRP'), ("'m", 'VBP'), ('happy', 'JJ'), ('Clap', 'NNP'), ('along', 'IN'), ('if', 'IN'), ('you', 'PRP'), ('feel', 'VBP'), ('like', 'IN'), ('a', 'DT'), ('room', 'NN'), ('without', 'IN'), ('a', 'DT'), ('roof', 'NN')]
# Ver significado das tags em https://pythonprogramming.net/part-of-speech-tagging-nltk-tutorial/

print('sentiment:',text.sentiment)
# sentiment: Sentiment(polarity=0.8, subjectivity=1.0)

print(text.sentences[0].words[1])
# went

print(text.sentences[0].words[1].lemmatize("v"))
# go


