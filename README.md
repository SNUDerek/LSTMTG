# mtgcardgenerator

[WIP] character-based recurrent langauge model for mtg card generation based on RoboRosewater

## purpose

generate MtG cards using a character-based language model. potential developments include seq-2-seq model for generating card based on name (and accompanying name generator), and a model that will generate each part of the card individually or in a hierarchical manner.

## idea

each card is processed into a sequence of characters (special symbols are encoded as unique unicode characters, reminder text is removed, card name in body text is replaced with a single token, and each section of the card [name, CMC, rarity etc] is separated with a divider character '|'). a recurrent language model is trained to predict the next chracter in a sequence based on the previous *w* characters, where *w* is the `window size`. the recurrent state is preserved over each card (so that besides the window-sized context characters, the network also 'remembers' the entire sequence thus far in its state), and reset between cards. during training, teacher forcing is used - instead of feeding the recurrent network its previous prediction, the true value is fed in.

at prediction, the network is given a small sequence of start-of-card tokens (and optionally one or more characters) and is allowed to predict a sequence of characters. unlike training, on prediction, the previous *predicted* character is fed in. this prediction is allowed to continue for a predefined number of characters.

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
3. run `keras_stateful_LM-v2.ipynb` to train model
4. run `keras_stateful_LM-v2decode.ipynb` to decode

some sample trained models are included.

generation code is available in the last notebook; it allows starting with a specific character sequence and adjusting softmax temperature to adjust the 'confidence' of the network

## results

here are some random results from 1~3 epochs:

```
# Ⓝ represents the title creature's name
# C, U, R, etc represent rarity (common, uncommon, rare...)

bringay of be
①Ⓤ
C
creature
elemental
flying
sorc any cards from your graveyard to its owner's library. if you do, then shuffle your library.
2/1

wrime
②Ⓤ
C
creature - horror
flying
when Ⓝ enters the battlefield and put a spell or ability counter and of turn.
2/1

ringer of ration
①Ⓤ
U
creature - human wizard
when Ⓝ enters the battlefield, return the bick this turn, you gain 1 life.
1/1

aldon callin
②Ⓦ
R
sorcery
exile your library.


aren shrane
①Ⓑ
U
creature - human wizard
when Ⓝ enters the battlefield, you may perm.
1/1


pubblen chalte
①Ⓑ
C
creature - human wizard
flying
whenever a card from combat tramples. if you do, return target creature card from the top and warriord.
5/3


traun the of dick cavoter
②
C
creature - human wizard
when Ⓝ enters the battlefield, exile that cards from the number of from the basic land instead.
2/3
```




