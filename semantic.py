import spacy
nlp = spacy.load('en_core_web_md')
word1 = nlp("cat")
word2 = nlp("monkey")
word3 = nlp("banana")
print("First similarity")
print(word1.similarity(word2))
print(word3.similarity(word2))
print(word3.similarity(word1))

# Q1: Write a note about what you found interesting about the similarities between cat, monkey and banana and think of an example of your own.

''' A1: 
    On the similarity, you get a floating point number (percentage) of how similar two words are. If we look 
    at these percentages we see that cat and monkey are most similar with 59% because they are both animals. 
    Banana and monkey have a higher similarity then cat and banana, so the model seems to know that monkeys eat bananas.
'''
word1 = nlp("elephant")
word2 = nlp("peanuts")
word3 = nlp("mouse")
print("Own Example")
print(word1.similarity(word2))
print(word3.similarity(word2))
print(word3.similarity(word1))



''' Q2:
Run the example file with the simpler language model ‘en_core_web_sm’
and write a note on what you notice is different from the model
'en_core_web_md'.'''

'''A2: 
    Firstly I get a warning that the "The model you're using has no word vectors loaded" in the small model. 
    Also with the medium model the percentages are higher on the comparissons with the complaints to the recipes. 

'''
tokens = nlp('cat apple monkey banana ')
print("\nSecond similarity")
for token1 in tokens:
    for token2 in tokens:
        print(token1.text, token2.text, token1.similarity(token2))


sentence_to_compare = "Why is my cat on the car"
sentences = ["where did my dog go",
"Hello, there is my car",
"I\'ve lost my car in my car",
"I\'d like my boat back",
"I will name my dog Diana"]

print("\nThird similarity")
model_sentence = nlp(sentence_to_compare)
for sentence in sentences:
    similarity = nlp(sentence).similarity(model_sentence)
    print(sentence + " - ", similarity)

