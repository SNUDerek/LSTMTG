# mtgcardgenerator

[WIP] character-based recurrent langauge model for mtg card generation based on RoboRosewater

## purpose

generate MtG cards using a character-based language model.

## idea

each card is processed into a sequence of characters (special symbols are encoded as unique unicode characters, reminder text is removed, card name in body text is replaced with a single token, and each section of the card [name, CMC, rarity etc] is separated with a divider character '|'). a recurrent language model is trained to predict the next chracter in a sequence ~~based on the previous *w* characters, where *w* is the `window size`~~ given what it has generated before for *this card*. the recurrent state is preserved over each card (so the network 'remembers' the entire sequence thus far in its state), and reset between cards. during training, teacher forcing is used - instead of feeding the recurrent network its previous prediction, the true value is fed in.

at prediction, the network is given a start-of-card token and is allowed to predict a sequence of characters. unlike training, on prediction, the previous *predicted* character is fed in. this prediction is allowed to continue for a predefined number of characters.

## network architecture

at each timestep, (at least) one input is sent to the network. by combining the information of the previous input and the state, the network can predict the next output. After each batch of *n* cards (here, *n* = 1), the state is reset. Here we use 'teacher forcing' by inputting the true input and not the network's previous output.

```
# network diagram:

					  E     << input t
					  |
 ... -> [ state_c ] -> [ state_d ] -> [state_e ]
 					  |
					  F     << output t+1
```


## requirements
```
h5py
keras
pandas
tensorflow
```

## background and previous work

### roborosewater post on mtgsalvation:

http://www.mtgsalvation.com/forums/magic-fundamentals/custom-card-creation/612057-generating-magic-cards-using-deep-recurrent-neural

### `keras` stateful LSTM and recurrent LM tutorials:

https://blog.keras.io/a-ten-minute-introduction-to-sequence-to-sequence-learning-in-keras.html

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

1. run `00_datareader.ipynb` to read cards from json
2. run `01_dataformatter.ipynb` to process data
3. run `02_keras_LM_train.ipynb` to train model
4. run `03_keras_LM_decode.ipynb` to generate

some sample trained models are included.

generation code is available in the last notebook; it allows starting with a specific character sequence and adjusting softmax temperature to adjust the 'confidence' of the network. if the temperature is too low (0.1), the network can be overly conservative in its guesses and converge on repeating strings of ("of of of of of of of of") which should make sense to any MtG player given the number of cards with X of Y as a name. below around 0.1, there can be underflow/divide-by-zero errors due to the implementation of temperature. if the temperature is too high (near 1.0), the model can be too flexible in its guessing, resulting in a lot of non-word jibberish.

## results

here are some random results from 1~3 epochs:

```
# Ⓝ represents the title creature's name
# C, U, R, etc represent rarity (common, uncommon, rare...)

singing star
①
U
artifact
↷: add ⒸⒸⒸ.

istor mentor
③Ⓖ
C
creature
human
shaman
2
3
ⓊⓊ, ↷: istor mentor deals damage equal to the number of +1/+1 counters on it to any target.

spirtort
ⓊⓊ
C
creature
bird
monk
2
2
flying
↷: target creature gets +2/-1 and gains intimidate until end of turn.

warmingpear's rapler
⑦
C
artifact
creature
hydra
4
2
ⒼⒼ: target creature loses double strike until end of turn.

sadking mistrakity
③Ⓦ
C
artifact
cumulative upkeep ①
sadking mistrakity can't block.
⑤Ⓖ, pay 2 life: prevent the next 5 damage that would be dealt to target creature or player this turn instead.
spells you cast cost ③ less to cast.
goblins you cast have minotaur 1.

oning-archive
③Ⓦ
U
creature
angel
3
3
flying
②: target creature loses all abilities until your next upkeep.

salker of the stand
②ⓊⓊ
R
creature
human
wizard
2
2
whenever salker of the stand becomes blocked, you may pay ②. if you do, put a +1/+1 counter on salker of the stand.

illanter's spike
②Ⓖ
R
sorcery
create a 1/1 white knight creature token with haste. exile it at the beginning of the next end step.
```




