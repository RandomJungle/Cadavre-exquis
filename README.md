# Writing models

## Char-level text generation

First version of the model is at char level, trained on the gutenberg corpus.

First step was writing char to int and int to char encodings to json. It was done prior to actual
training, in order to have a fixed reference of characters. The total number of characters was 
extracted from the gutenberg corpus in nltk, which resulted in a total of ***97 unique characters*** 
in total. 

Here is the first input of the functional model I trained. My input is in 
<span style="color:SteelBlue">blue</span> with the model participation in 
<span style="color:DarkSlateBlue">dark blue</span> :

>Let's play a game. You start by writing a first sentence :
> 
> <span style="color:SteelBlue">
> There once was a boy named kaki who loved to play hide and seek with his friends
> </span>
> <span style="color:DarkSlateBlue"> that in the
> seases and such a moment of their place of the turning
> enough to the little signess again.
> </span>
> <span style="color:SteelBlue">
> But his friends did not like the game that much and so, he seeked new friends to play with
> </span>
> <span style="color:DarkSlateBlue">
> the first side
> of the Barton.
> </span>
> <span style="color:SteelBlue">
> So he went into a deep forest where strange creatures lived
> </span>
> <span style="color:DarkSlateBlue  "> at the songevout of the canming but the
> deep in the white and the comman of his side, the same life, and he paid her the dispanites
> of the great close to be seen her beauty to be such a cost as the moderh and the whole way to all the rest
> steep, and there was not on the particular harms of the
> short and declared the boat.
> </span>
> <span style="color:SteelBlue">
> That was a sad game of hide and seek. THE END
> </span>

Being trained on classic english literature, the model output is interesting, but not so adapted to
the narrative side I wanted the game to have. I think a key element of this game is when its played
within the context of a story, because then every addition is taking the character into one direction, 
then another, which can result in an interesting collective building of a story.

So maybe retraining the model on fairytale story is a good option, to leverage that more narrative 
quality of language.

## Word-level text generation

Second version is at word level, trained on a corpus of fairy tales, with the assumption that it would
take more of a narrative style.