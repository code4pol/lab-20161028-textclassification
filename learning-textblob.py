from textblob.classifiers import NaiveBayesClassifier
from textblob import TextBlob

treinamento = [ ('Eu sou lindo', 'A'),
				('Eu sou feio','B')]

c = NaiveBayesClassifier(treinamento)

print(c.classify('seu feio'))
print(c.classify('seu horroroso'))
print(c.classify('seu bonito'))
print(c.classify('seu lindo'))

print('labels:',c.labels())

test = [ ('Voce e muito gato','A'),
		 ('Voce e muito feio','B')]

print('acuracia:',c.accuracy(test))

test = [ ('Voce e muito lindo','A'),
		 ('Voce e muito feio','B')]

print('acuracia:',c.accuracy(test))

print('features:',c.extract_features('Eu sou horroroso'))
print('informative:',c.show_informative_features())

# NLP
text = TextBlob("I went home. Because I'm happy. Clap along if you feel like a room without a roof.")
print('text:',text)
print('sentences:',text.sentences)
print('words:',text.words)
print('tags:',text.tags)
print('sentiment:',text.sentiment)

print(text.sentences[0].words[1])
print(text.sentences[0].words[1].lemmatize("v"))






twitter.com/Rafael54992115/status/721341797801279489
twitter.com/magnesio/status/721056029631520768


