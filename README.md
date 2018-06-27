# mtgcardgenerator

[WIP] character-based recurrent language model for mtg card generation inspired by RoboRosewater

## purpose

generate MtG cards using a character-based language model. for science.

## requirements

```
h5py
numpy
pandas
keras==2.1.6
tensorflow==1.7.0
```

while `keras` can support multiple backends, i am using the `tensorflow` backend with CUDA support (`tensorflow-gpu`).

## main idea

each card is processed into a sequence of characters (special symbols are encoded as unique unicode characters, reminder text is removed, card name in body text is replaced with a single token, and each section of the card [name, CMC, rarity etc] is separated with a unique divider character). a recurrent language model is trained to predict the next character in a sequence given what it has generated before for *this card*. the recurrent state is preserved over each card (so the network 'remembers' the entire sequence thus far in its state), and is reset between cards. during training, teacher forcing is used - instead of feeding the recurrent network its previous prediction, the true value is fed in.

at prediction, the network is given a start-of-card token and is allowed to predict a sequence of characters. unlike training, on prediction, the previous *predicted* character is fed in. we randomly sample the character according to the network's output *probability distribution* rather than choosing the top-1 most greedy prediction, in order to allow some 'creativity', and we can modulate the sampling using the *temperature*, which adjusts the 'confidence' of the prediction (higher temperature = flatter distribution = more 'creativity'; lower temperature >= 1.0 = more conservative predictions). this prediction is allowed to continue for a predefined number of characters or until we reach an end-of-sequence tag.

## network architecture

the recurrent neural network generates an output `h_i` at each timestep `i` as a function of the previous state `h_i-1` and the current input `x_i`. when this `h_i`. the *Long Short-Term Memory* adds another memory state `C` as well as the output state `H`. essentially, this means that we are training the model to consider both the previous information (in `c_i-1` and `h_i-1`) as well as the current input `x_i` in order to predict the next item (in this case; in e.g. a *sequence labeling* task, we would use the LSTM output to predict the current *label* for input `x_i`).

at each timestep, (at least) one input is sent to the network. by combining the information of the previous input and the state, the network can predict the next output. After each batch of *n* cards (here, *n* = 1), the state is reset randomly, and the next sequence begins with a 'beginning-of-sequence' tag to tell the network that we are beginning a new card. Here we use 'teacher forcing' by inputting the true input and not the network's previous output.

```
# network diagram:

					  E     << input t
					  |
 ... -> [ state_c ] -> [ state_d ] -> [state_e ]
 					  |
					  F     << output t+1
```

the current model, which works well, uses a character embedding size of `500` and two LSTM layers of `1000` features each, and a maximum sequence length of 256, which captures approximately 95% of the cards (minus flip-cards and planeswalkers). *Dropout regularization* of 0.4 is used at multiple points to force the network to generalize better. because teacher forcing is used, the training model takes all 256 inputs at once (this is a `keras` thing), and any inputs past the end of the card are 'masked' so that they don't affect the backpropagation = the network does not learn from them. at decode, the LSTM weights are loaded into a single-timestep model which is allowed to generate probabilities one at a time.

```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
lm_input (InputLayer)        (None, 256)               0         
_________________________________________________________________
lm_emb (Embedding)           (None, 256, 500)          51000     
_________________________________________________________________
dropout_1 (Dropout)          (None, 256, 500)          0         
_________________________________________________________________
lm_lstm1 (LSTM)              [(None, 256, 1000), (None 6004000   
_________________________________________________________________
dropout_2 (Dropout)          (None, 256, 1000)         0         
_________________________________________________________________
lm_lstm2 (LSTM)              [(None, 256, 1000), (None 8004000   
_________________________________________________________________
dropout_3 (Dropout)          (None, 256, 1000)         0         
_________________________________________________________________
lm_dns_1 (Dense)             (None, 256, 1000)         1001000   
_________________________________________________________________
dropout_4 (Dropout)          (None, 256, 1000)         0         
_________________________________________________________________
lm_dns_final (Dense)         (None, 256, 102)          102102    
=================================================================
Total params: 15,162,102
Trainable params: 15,162,102
Non-trainable params: 0
_________________________________________________________________
```

an alternative approach is to make a *stateless* RNN that takes a long sequence of characters, and outputs its prediction of the next character; for example, it may read 100 characters, and guess the next character, then the window is moved over one step to include the previous character, and the next character is then generated. this is not an uncommon implementation, and can be used in, for example, networks that generate random texts. the benefit of this system is that the states are reset each time, preventing saturation of the cell states (vanishing gradient), which is useful for long text generation because it may be hard to estimate when to manually reset cells. however, because MtG cards are so short, in order to make text long enough, multiple cards must be strung together, which may 'confuse' the network into learning false dependencies between cards (the end of one card might affect the predictions of the next). since card text is short and there is a clear beginning and end point, this method seems more effective (the other method was experimented with and the results were less coherent), and we do not have to 'seed' the LSTM with, say, 99 random charcaters plus the start-of-sequence tag to generate the first real character we want.

## extensions

at decode, we can also 'force' certain subsequences. for example, forcing a card name (using the `seed` parameter) is demonstrated. but with the recent implementation of field-specific dividers that allow us to parse the card more easily, we are also provided a way to insert information *into* the sequence at decoding. we can use the teacher forcing method to effectively 'overwrite' any subsection with the desired information once we see the network generate the appropriate start tag. for example, if we want to force type *creature*, can wait until we see the `type` tag appear, then instead of feeding in the predictions, feed in the sequence `c, r, e, a, t, u, r, e, <begin-P/T>` to force the generation of a creature. 

## to run

1. run `00_datareader.ipynb` to read cards from json
2. run `01_dataformatter.ipynb` to process data
3. run `02_keras_LM_train.ipynb` to train model
4. run `03_keras_LM_decode.ipynb` to generate

~~some sample trained models are included.~~

generation code is available in the last notebook; it allows starting with a specific character sequence and adjusting softmax temperature to adjust the 'confidence' of the network. if the temperature is too low (0.1), the network can be overly conservative in its guesses and converge on repeating strings of ("of of of of of of of of") which should make sense to any MtG player given the number of cards with X of Y as a name. below around 0.1, there can be underflow/divide-by-zero errors due to the implementation of temperature. if the temperature is too high (near 1.0), the model can be too flexible in its guessing, resulting in a lot of non-word jibberish.

the model shown was trained on the following setup:  

```
GPU: Nvidia GTX 1060 6GB  
CPU: Intel i7-5820K  
RAM: 48 GB
OS : Ubuntu 16.04
ETC: python 3.6 (anaconda), jupyter lab
```

## results

here are some random results from 10 epochs:

```

Derate The Scrapper
②ⒼⓊ
rare
creature: beast
when Derate The Scrapper enters the battlefield, target opponent creates a 4/4 white knight creature token with trample.
4/3

Rooth
②Ⓡ
common
creature: beast
Ⓡ: Rooth gets +1/+0 until end of turn.
3/1

Castauxtlate
②ⒷⒷ
uncommon
sorcery
destroy target land.

Tather Tap
③
uncommon
artifact
at the beginning of each player's upkeep, that player sacrifices an artifact, creature, or land. that player loses that much life.

Dator Defonders
③Ⓖ
rare
creature: elf advisor
whenever you cast an artifact spell, counter that spell.
2/3

Toosor
②Ⓡ
rare
creature: human rogue
Ⓖ, ↷: draw a card.
evoke ①ⓌⓌⓌ
3/3
```

## background, previous work, and references

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


