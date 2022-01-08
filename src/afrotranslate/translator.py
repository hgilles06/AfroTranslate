from joeynmt.helpers import load_config, load_checkpoint
from joeynmt.vocabulary import Vocabulary
from joeynmt.model import build_model
from joeynmt.batch import Batch
from joeynmt.data import MonoDataset
from torchtext.legacy.data import Field # pylint: disable=no-name-in-module
from joeynmt.constants import UNK_TOKEN, PAD_TOKEN, EOS_TOKEN
from joeynmt.prediction import parse_test_args, validate_on_data

from typing import Dict

import os
import torch

from google_drive_downloader import GoogleDriveDownloader as gdd
import pandas as pd
import pkg_resources

import warnings
from spacy.lang.en import English # updated





class MasakhaneTranslate:

        def __init__(self, model_name:str, version:str=None,  device: str = 'cpu'):
            """
            model_name: name of the model. See the list of the models (directory names) from https://github.com/masakhane-io/masakhane-mt/tree/master/benchmarks
            version: most of the models have several versions. Each version is provided as a subdirectory of the model's directory from https://github.com/masakhane-io/masakhane-mt/tree/master/benchmarks
            device: device to use for inference ("cpu" or "cuda")
            """

            self.model, self.cfg, self.src_vocab, self.use_cuda = self.load_model(model_name, version=version, device=device)

            """
    	    cfg: configuration dictionary
    	    batch_class: class type of batch
    	    """


        def download_model(self, model_name:str):
            print("Downloading", model_name, "...")
            print(os.getcwd())

            links_models_path = pkg_resources.resource_filename('afrotranslate', 'links_models.csv')

            df = pd.read_csv(links_models_path)

            try:
                link = df.loc[df.model_name==model_name, "link"].values[0]
            except:
                raise ValueError("Model does not exist. Please select between the following list:", list(df.model_name))

            id = link.split('/')[-2]


    	    #dest_path = pkg_resources.resource_filename(f'afrotranslate.models.{model_name}', f'{model_name}.zip')
            dest_dir = pkg_resources.resource_filename(f'afrotranslate', 'models')

            os.makedirs(dest_dir+f"/{model_name}")
            dest_path = dest_dir+f"/{model_name}/{model_name}.zip"
            gdd.download_file_from_google_drive(file_id=id,
    		                            dest_path=dest_path,
    		                            unzip=True)


            os.remove(dest_path)

            print(model_name, "downloaded!")


        def load_model(self, model_name:str, version:str=None, device:str="cpu") -> torch.nn.Module:

            dest_dir = pkg_resources.resource_filename(f'afrotranslate', 'models')
            model_dir = dest_dir+f"/{model_name}"

            if not os.path.isdir(model_dir):
                self.download_model(model_name)

            if (version is None)or(version==""): #or(not os.path.isdir(model_dir+"/"+version)):
                version = os.listdir(model_dir)[0]
                print("As you don't provide any version we use this one by default:", version)
                print("Here is the complete list of versions:", os.listdir(model_dir))

            if (not version in os.listdir(model_dir)) : #subdir not in directory
                first_element_in_dir = os.listdir(model_dir)[0]
                if os.path.isdir(model_dir+"/"+first_element_in_dir):
                    raise ValueError('This version does not exit. Please select between the following list:', os.listdir(model_dir))
                else:
                    warnings.warn("There is only one version for this model!")


            if not os.path.isdir(model_dir+"/"+version): #if there is no subdirectory
                version=""


            model_dir = model_dir+"/"+version
            cfg_file = model_dir+"/config.yaml"
            ckpt=model_dir+"/model.ckpt"

            cfg = load_config(cfg_file)


            # read vocabs
            src_vocab_file = model_dir+ "/" +cfg["data"]["src_vocab"]
            trg_vocab_file = model_dir + "/" +cfg["data"]["trg_vocab"]
            src_vocab = Vocabulary(file=src_vocab_file)
            trg_vocab = Vocabulary(file=trg_vocab_file)

            if device is None:
                use_cuda = torch.cuda.is_available()
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            if device =="cpu":
                use_cuda = False

            # load model state from disk
            model_checkpoint = load_checkpoint(ckpt, use_cuda=use_cuda)

            # build model and load parameters into it

            model = build_model(cfg["model"], src_vocab=src_vocab, trg_vocab=trg_vocab)
            model.load_state_dict(model_checkpoint["model_state"])
            model.to(device)

            return model, cfg, src_vocab, use_cuda


        def translate(self, src_input:str, n_best:int=1) -> str:

            """
            Code inspired by "prediction.py" from JoeyNMT

            src_input: sentence in the source language
            n_best: amount of candidates to display. Limited by the beam size in the configuration file.
            """

            def _load_line_as_data(line):
                """ Create a dataset from one line via a temporary file. """
        		# write src input to temporary file
                tmp_name = "tmp"
                tmp_suffix = ".src"
                tmp_filename = tmp_name+tmp_suffix
                with open(tmp_filename, "w", encoding="utf-8") as tmp_file:
                    tmp_file.write("{}\n".format(line))

                test_data = MonoDataset(path=tmp_name, ext=tmp_suffix,
        		                        field=src_field)

        		# remove temporary file
                if os.path.exists(tmp_filename):
                    os.remove(tmp_filename)

                return test_data

            def _translate_data(test_data):
                """ Translates given dataset, using parameters from outer scope. """
                score, loss, ppl, sources, sources_raw, references, hypotheses, \
                hypotheses_raw, attention_scores = validate_on_data(
        		    self.model, data=test_data, batch_size=batch_size,
        		    batch_class=Batch, batch_type=batch_type, level=level,
        		    max_output_length=max_output_length, eval_metric="",
        		    use_cuda=self.use_cuda, compute_loss=False, beam_size=beam_size,
        		    beam_alpha=beam_alpha, postprocess=postprocess,
        		    bpe_type=bpe_type, sacrebleu=sacrebleu, n_gpu=n_gpu, n_best=n_best)
                return hypotheses


            data_cfg = self.cfg["data"]
            level = data_cfg["level"]
            lowercase = data_cfg["lowercase"]

            tok_fun = lambda s: list(s) if level == "char" else s.split()

            src_field = Field(init_token=None, eos_token=EOS_TOKEN, pad_token=PAD_TOKEN,
                          tokenize=tok_fun, batch_first=True, lower=lowercase,
                          unk_token=UNK_TOKEN, include_lengths=True)
            src_field.vocab = self.src_vocab

            # parse test args
            batch_size, batch_type, _, device, n_gpu, level, _, \
            max_output_length, beam_size, beam_alpha, postprocess, \
            bpe_type, sacrebleu, _, _ = parse_test_args(self.cfg, mode="translate")


            #Sentence tokenizing: useful in case there are several sentences
            nlp = English()
            nlp.add_pipe('sentencizer')
            doc = nlp(src_input)
            sentences = [sent.text.strip() for sent in doc.sents]


            if len(sentences)==1:
                # every line has to be made into dataset
                test_data = _load_line_as_data(line=src_input)
                hypotheses = _translate_data(test_data)[:n_best]
                return hypotheses[0] if n_best==1 else hypotheses

            print("Several sentences are detected. We split and translate them sequentially :).")
            hypotheses_dictionary = {}
            for i,sentence in enumerate(sentences):
                # every line has to be made into dataset
                test_data = _load_line_as_data(line=sentence)
                hypotheses_dictionary[f"Sentence{i+1}"] = _translate_data(test_data)[0] if n_best==1 else _translate_data(test_data)[:n_best]

            return hypotheses_dictionary
