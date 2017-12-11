# mtgcardgenerator

[WIP] character-based recurrent langauge model for mtg card generation based on RoboRosewater

## purpose

generate MtG cards using a character-based language model. potential developments include seq-2-seq model for generating card based on name (and accompanying name generator), and a model that will generate each part of the card individually or in a hierarchical manner.

## requirements
```
h5py
keras
pandas
tensorflow
tqdm
```

## background and previous work

### roborosewater post on mtgsalvation:
http://www.mtgsalvation.com/forums/magic-fundamentals/custom-card-creation/612057-generating-magic-cards-using-deep-recurrent-neural

### `keras` stateful LSTM and recurrent LM tutorials:

http://philipperemy.github.io/keras-stateful-lstm/

https://machinelearningmastery.com/understanding-stateful-lstm-recurrent-neural-networks-python-keras/

also see 파이썬 딥러닝 케라스 by 김태영

### other references

**softmax temperature in `keras`**

https://github.com/fchollet/keras/issues/3092

https://stackoverflow.com/questions/37246030/how-to-change-the-temperature-of-a-softmax-output-in-keras/37254117#37254117

### card data

the card JSON file is from https://mtgjson.com

## to run

1. run `datareader.ipynb` to read cards from json
2. run `dataformatter.ipynb` to process data
3. run `keras_stateful_LM.ipynb` to train model

generation code is available in the last notebook; it allows starting with a specific character sequence and adjusting softmax temperature to adjust the 'confidence' of the network

## results

TBD: only small trial networks have been tried. some sample cards can be found in `/samples` (made with https://mtgcardsmith.com)




