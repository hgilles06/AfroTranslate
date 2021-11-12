# AfroTranslate

This package allows you to obtain translations from Masakhane JoeyNMT based models with very few lines of code. Masakhane is a grassroots research community aiming to revive and strengthen African languages through AI.

Available models can be found [here](https://github.com/masakhane-io/masakhane-mt/tree/master/benchmarks).

Note: Please, install the cuda supported version of pytorch to use the GPU. Ex: pip install torch==1.8.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html.

# Installation
pip install AfroTranslate

# Example

from afrotranslate import MasakhaneTranslate

## Translation using the English to Fon model
translator = MasakhaneTranslate(model_name="en-fon")

translator.translate("I love you so much!", n_best=1)

'Un yí wǎn nú we tawun'
