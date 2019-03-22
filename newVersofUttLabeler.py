import numpy as np
import re
from keras.models import Sequential
from keras.models import Model
from keras import backend as K
from keras.layers import Dense
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras import backend as K
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.models import load_model

filePath = r'/home/nikki/2012-10-20-coding_data-all.txt'
word2VecPath = r'/home/nikki/vec.txt'

def normalize_code(code):
    #    print(code,spk)
    code = re.sub(r"(.*)([+-]).*", r"\1\2", code).upper()
    if code in ["FA", "GI", "RES", "QUC", "QUO", "REC"]:
        pass
    elif code in ["ADW", "CO", "DI", "RCW", "WA"]:
        code = "NA"
    else:
        code = "COU"
    return code

#assign every word an index
def wordToIndex(open_path):
    with open(open_path, 'r') as file:
        count = 0
        wordIndexDict = {}
        for line in file:
            values = line.split('\t')
            if len(values) == 10:
                if values[6] == 'T':
                    result = values[-1]
                    for word in result.split(' '):
                        if word in wordIndexDict:
                            continue
                        else:
                            count = count + 1
                            wordIndexDict[word] = count
    return wordIndexDict

#convert each utterance into a vector of indices (add zeros if the vector is less than 20 words, otherwise cut it off)
def fileToUtteranceAndIndices(open_path, wordIndexDict):
    transcriptToUtterances = {}
    with open(open_path, 'r') as file:
        for line in file:
            values = line.split('\t')
            if len(values) == 10:
                if values[6] == 'T':
                    result = values[-1]
                    vec = []
                    for word in result.split(' '):
                        vec.append(wordIndexDict[word])
                    label = str(values[5])
                    label_code = str(normalize_code(label))
                    if label_code == "FA":
                        label_code = [1, 0, 0, 0, 0, 0, 0, 0]
                    elif label_code == "GI":
                        label_code = [0, 1, 0, 0, 0, 0, 0, 0]
                    elif label_code == "RES":
                        label_code = [0, 0, 1, 0, 0, 0, 0, 0]
                    elif label_code == "QUC":
                        label_code = [0, 0, 0, 1, 0, 0, 0, 0]
                    elif label_code == "QUO":
                        label_code = [0, 0, 0, 0, 1, 0, 0, 0]
                    elif label_code == "REC":
                        label_code = [0, 0, 0, 0, 0, 1, 0, 0]
                    elif label_code == "NA":
                        label_code = [0, 0, 0, 0, 0, 0, 1, 0]
                    else:
                        label_code = [0, 0, 0, 0, 0, 0, 0, 1]
                    if len(vec) < 20:
                        zeroes = 20 - len(vec)
                        for zero in range(zeroes):
                            vec.append(0)
                        add_pair = [vec, label_code]
                    elif len(vec) > 20:
                        newVec = vec[0:20]
                        add_pair = [newVec, label_code]
                    else:
                        add_pair = [vec, label_code]
                    key = values[0]
                    if key in transcriptToUtterances:
                        transcriptToUtterances[key].append(add_pair)
                    else:
                        transcriptToUtterances[key] = [add_pair]
    return transcriptToUtterances

#currently not AVERAGE VECTORS, they are WORD INDICES from a WORDTOINDEX DICT with added ZEROS if under length 20
def collectUtterances(file2AvgVecPerUtterance):
    listOfAvgVectors = []
    listOfLabels = []
    for key in file2AvgVecPerUtterance:
        for pair in file2AvgVecPerUtterance[key]:
            listOfAvgVectors.append(pair[0])
            listOfLabels.append(pair[1])
    listOfAvgVectors = np.array(listOfAvgVectors)
    listOfLabels = np.array(listOfLabels)
    return listOfAvgVectors, listOfLabels

wordIndexDict = wordToIndex(filePath)
#print('word index dict: ', wordIndexDict)
utters = fileToUtteranceAndIndices(filePath, wordIndexDict)
#trans_to_utters = word2vec(utters)
np.random.seed(7)
# load data
X, Y = collectUtterances(utters)
#utters = make_dictionary(filePath)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
# create model
X_train = X_train[:200]
y_train = y_train[:200]

model = Sequential()
model.add(Embedding(11864, 100, input_length=20))
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(60, activation='relu'))
model.add(Dense(40, activation='relu'))
#I need to get this layer, feed it into another lstm and somehow get an empathy rating
model.add(Dense(8, activation='softmax'))
# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit the model
history = model.fit(X_train, y_train, epochs=1, batch_size=10)
# evaluate the model
scores = model.evaluate(X_test, y_test)
# list all data in history
argmaxModel = model.predict(X_test)

inter  = K.function([model.layers[0].input, K.learning_phase()], [model.layers[3].output])

##### transcript processing
final_data
for trans in transcrit.keys():
    utter = data[trans]

    inter_output = inter([procc_utter,0])[0]
    avg = np.mean(ss)
    final[transid] =

print(inter_output)

argmaxModel2 = np.argmax(argmaxModel, axis=-1)
argmaxModelYTEST = np.argmax(y_test, axis=-1)
a = np.array(argmaxModel2)
b = np.zeros((31516, 8))
#y true
b[np.arange(31516), a] = 1
c = np.array(argmaxModelYTEST)
d = np.zeros((31516, 8))
#y pred
d[np.arange(31516), c] = 1
print('train loss', history.history['loss'][-1])
print('test loss', scores[0])
print(classification_report(b, d))
model.save('my_model.h5')  # creates a HDF5 file 'my_model.h5'
del model  # deletes the existing model
raise Exception
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('accuracy.png')
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('loss.png')
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
print(model.summary())
model.save('/home/nikki/utteranceNN.weights')
plot_model(model, toc_file='model.png')