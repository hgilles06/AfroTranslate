# AfroTranslate

This package allows you to obtain translations from Masakhane JoeyNMT based models with very few lines of code. Masakhane is a grassroots research community aiming to revive and strengthen African languages through AI.

Available models can be found [here](https://github.com/masakhane-io/masakhane-mt/tree/master/benchmarks).

Note: Please, install the cuda supported version of pytorch to use the GPU. Ex: pip install torch==1.8.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html.


Here are some useful links:

[Masakhane](https://www.masakhane.io/) Visit the Masakhane home page: 

[Masakhane web](http://translate.masakhane.io/): Get translations using the Masakhane web application.

[Masakhane Github](https://github.com/masakhane-io): Take a look at the community main projects here!

[JoeyNMT Github](https://github.com/joeynmt/joeynmt): Get hands on JoeyNMT here!

# Test it right now on [Colab](https://colab.research.google.com/drive/1_iqv6EMZ76Pkvmtb94ZxgDfIPGeaMP7-?usp=sharing)!

# Installation
pip install AfroTranslate

# Example:  

## Translation using the English to Fon model

from afrotranslate import MasakhaneTranslate

translator = MasakhaneTranslate(model_name="en-fon")

translator.translate("I love you so much!", n_best=1)

'Un yí wǎn nú we tawun'

## Translating several sentences at once: 

translator.translate("I love you so much! Our love is very strong!", n_best=1)

{'Sentence1': 'Un yí wǎn nú we tawun',
 'Sentence2': 'Wanyiyi mǐtɔn ɖò taji tawun'}


#### Disclaimer: 

This is a community research project and as such, this service is not a production system. The models are only trained using religious data. Therefore, it should not be used for official translations. 
