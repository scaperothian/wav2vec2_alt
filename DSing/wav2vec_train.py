import os
import sys
import time
import glob
import re
import json
import random
import tqdm

import torch
import torchaudio
import evaluate

import pandas as pd
import numpy as np
import IPython

from IPython.display import display, HTML, Audio

from datasets import load_dataset, load_metric, ClassLabel, DatasetDict

from torch.utils.data import DataLoader

from torchaudio.models.decoder import ctc_decoder
from torchaudio.models.decoder import download_pretrained_files

from transformers import Wav2Vec2CTCTokenizer
from transformers import Wav2Vec2FeatureExtractor
from transformers import pipeline

from transformers import TrainingArguments, Trainer
from transformers.utils import logging

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

#
# Dataset unpacked and preprocessed into child folder dataset/
# DSing is around 1.9GB after being preprocessed.
#
tokens_file="./tokens.txt"
dataset_folder = "../sing_300x30x2/damp_dataset"



# Create Pipeline
#
# Other ideas to support optimizing this model: https://huggingface.co/docs/transformers/v4.18.0/en/performance
#
#model_checkpoint="facebook/wav2vec2-large-960h-lv60-self"
model_checkpoint="facebook/wav2vec2-base"

#
# ctc_loss_reduction used here: https://huggingface.co/blog/fine-tune-wav2vec2-english
# apparently for Wav2Vec2 Sum is default, but for nn.functional, mean is default.
# most examples i've seen have a loss that aligns with the mean setting.
# 
asr_pipeline = pipeline("automatic-speech-recognition", model=model_checkpoint, model_kwargs={"ctc_loss_reduction": 'mean'})
model = asr_pipeline.model

# 
# Gradient Accumulation
# Used in example here: https://huggingface.co/blog/fine-tune-wav2vec2-english
# Per the code: By default, non-reentrant checkpoint stops recomputation as soon as it
#               has computed all needed Tensors.
#
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

#
# Used in example here: https://huggingface.co/blog/fine-tune-wav2vec2-english
#
model.freeze_feature_encoder()

# Count the number of trainable parameters in the model
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Checkpoint: {model_checkpoint}.  Trainable parameters:", trainable_params)


# Note: I want to use a beam search library that has been pre-trained on librispeech.  the pretrained beam search 
#       has been created with all lowercase characters (i.e. phoneme settings, etc.).  I am changing the default 
#       model tokenizer vocab settings to accept lower case inputs so i can use the decoder.  However, it will also
#       accept upper case, so this may also impact performance but only for inference.  (i.e. not sure where this 
#       is normalized to .
asr_pipeline.tokenizer.do_lower_case = True
target_sampling_rate = asr_pipeline.feature_extractor.sampling_rate

files = download_pretrained_files("librispeech-4-gram")
# Found from Fairseq for Wav2Vec2 - 
# https://github.com/facebookresearch/fairseq/blob/main/examples/wav2vec/config/finetuning/vox_960h_2_aws.yaml
# Note: I am not using the same lexicon file though...
# LM_WEIGHT = 2.0
# WORD_SCORE = 0
# SIL_SCORE = -1

LM_WEIGHT = 3.23
WORD_SCORE = -0.26
SIL_SCORE = 0

beam_search_decoder = ctc_decoder(
    lexicon=files.lexicon,
    tokens=tokens_file,
    lm=files.lm,
    nbest=1,
    beam_size=512,
    lm_weight=LM_WEIGHT,
    word_score=WORD_SCORE,
    sil_score=SIL_SCORE,
    blank_token='<pad>',
    unk_word='<unk>'
)

greedy_decoder = ctc_decoder(
    lexicon=files.lexicon,
    tokens=tokens_file,
    lm=files.lm,
    nbest=1,
    beam_size=1,
    lm_weight=LM_WEIGHT,
    word_score=WORD_SCORE,
    sil_score=SIL_SCORE,
    blank_token='<pad>',
    unk_word='<unk>'
)

wer_metric = evaluate.load("wer")

def compute_metrics_dummy(eval_pred):
    return {'wer':1.0}


def compute_metrics(eval_pred,kind='beam',compute=True):
    """
    Calculates WER for a batch of logits and their labels.

    eval_pred - tuple (logit output from the model, token labels from dataset)
    kind - can compare between beam search and greedy search.  both using kenlm 

    compute - bool - for training this will compute WER every time its logged.  
                     this is nice for understanding if the training is working.
                     for evaluation, this is set to false so compute is run after
                     all batches are processed.

    output is the WER computed from the batch.  if the model is run multiple times, the 
    batch WERs are aggregated.

    Note: add_batch and then doing compute will clear the previously cached batch results.
    """
    logits, labels = eval_pred
    #print(f"Logit Type: {type(logits)}")

    # In some scenarios, the input the compute_metrics is a tensor.
    if type(logits) is np.ndarray:
        logits = torch.Tensor(logits)
    else:
        # copy this tensor for computing things...
        logits = logits.clone().detach().requires_grad_(False)    
    #print(f"Changing Logit Type to: {type(logits)}")
    #print(f"{logits.shape}")
    
    if kind=='beam':
        # Creates a list of lists that are of size [batch_size,emissions,vocab_size]
        #
        # Where output[0][0] gives you the CTCHypothesis object.
        #
        # Extract transcript from output[0][0].words (i.e. list of words).  
        # May need to join depending on objective.
        #
        predictions = beam_search_decoder(logits)
    elif kind=='greedy':
        # Creates a list of lists that are of size [batch_size,1]
        #
        # Where output[0][0] gives you the CTCHypothesis object.
        #
        # Extract transcript from output[0][0].words (i.e. list of words).  
        # May need to join depending on objective.
        #
        predictions = greedy_decoder(logits)
    else:
        print(f"Error passing in decoder kind: {kind}")
        sys.exit()

    ref = asr_pipeline.tokenizer.batch_decode(labels)
    pred = [" ".join(prediction[0].words) for prediction in predictions]

    wer_metric.add_batch(predictions=pred, references=ref)

    if compute: 
        return {"wer":wer_metric.compute()}

    return None

def prepare_dataset(batch,tokenizer,feature_extractor):
    """
    Creating a new dataset with the map function to generate the 
    keys below.  Padding will occur in the data collator on a per
    batch basis. 

    Inputs (i.e. feature extractor):
    input_values   - tensor array for audio samples (shape=(n,) - where n is the number of audio samples)
    attention_mask - used for expressing where there are padded samples 

    Outputs (i.e. tokenizer related)
    labels - tensor array for text output tokens (i.e. not transcript).  (shape=(m,) - where m is the number of character tokens)
    """
    # Cleaning data...
    chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"]'
    batch["transcription"] = re.sub(chars_to_ignore_regex, '', batch["transcription"]).lower() + " "
    
    audio = batch["audio"]

    # Feature Extractor manipulation
    #
    # this object will return a list of lists because the 
    # transcriptions are not padded (i.e. as opposed to a 
    # Tensor of tensors when using return_tensors='pt').
    # Padding is done per batch to optimize the size for inference and 
    # training.
    #
    # data_collator is responsible for padding the data.
    inputs_values_pt = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"])
    
    if "attention_mask" in inputs_values_pt:
        batch["attention_mask"] = inputs_values_pt.attention_mask
        
    batch["input_values"] = inputs_values_pt.input_values[0]
    batch["input_length"] = len(batch["input_values"])
    
    # Tokenizer manipulation
    #
    # this object will return a list of lists because the 
    # transcriptions are not padded (i.e. as opposed to a 
    # Tensor of tensors when using return_tensors='pt').
    # Padding is done per batch to optimize the size for inference and 
    # training.
    #
    # data_collator is responsible for padding the data.
    labels_pt = tokenizer(batch["transcription"])
    batch["labels"] = labels_pt['input_ids']
    
    return batch

def show_random_elements(dataset, num_examples=10):
    assert num_examples <= len(dataset), "Can't pick more elements than there are in the dataset."
    picks = []
    for _ in range(num_examples):
        pick = random.randint(0, len(dataset)-1)
        while pick in picks:
            pick = random.randint(0, len(dataset)-1)
        picks.append(pick)

    df = pd.DataFrame(dataset[picks])

    # Used with Jupyter...
    display(HTML(df.to_html()))

@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the already tokenized inputs received.
    Args:
        processor (:class:`~transformers.Wav2Vec2Processor`)
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).

            Other Options in the pad method that are NOT implemented for this class (i.e. I always want to pad to longest for the 
            input and the labels)
            * (not implemented) :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * (not implemented) :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).

    Reference Code here:
    https://huggingface.co/blog/fine-tune-wav2vec2-english

    
    Note: in the example referenced above, there were parameters for padding max length, etc.  I have created some logic 
    in the prepare_dataset to support truncation of data for testing and benchmarking.  I do not think i need max_length 
    options for collator at this time.

    """
    
    tokenizer: Wav2Vec2CTCTokenizer
    feature_extractor: Wav2Vec2FeatureExtractor
    padding: Union[bool, str] = "longest"

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        
        # Features in this case is a list of batch size that contains DataSet objects from the train split
        # (including pretokenized labels). the output batch has been changed from a list back to a dictionary 
        # with the respective data objects.
        #
        # Note for future self: 
        # pad is being called from PreTrainedTokenizerBase.pad.  From docs:
        #      Pad a single encoded input or a batch of encoded inputs up to predefined length or to the max sequence length
        #      in the batch.
        #      
        #    Padding side (left/right) padding token ids are defined at the tokenizer level (with `self.padding_side`,
        #    `self.pad_token_id` and `self.pad_token_type_id`).

        #    Please note that with a fast tokenizer, using the `__call__` method is faster than using a method to encode the
        #    text followed by a call to the `pad` method to get a padded encoding.
        # 
        #         <Tip>
        # 
        #         If the `encoded_inputs` passed are dictionary of numpy arrays, PyTorch tensors or TensorFlow tensors, the
        #         result will use the same type unless you provide a different tensor type with `return_tensors`. In the case of
        #         PyTorch tensors, you will lose the specific device of your tensors however.
        # 
        #         </Tip>

        # Audio Input Data (not tokenized)
        input_features = [{"input_values": feature["input_values"]} for feature in features]

        # batch is a dictionary-like type.
        batch = self.feature_extractor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )
        
        # Tokenized Transcript Labels (character level tokens)
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.tokenizer.pad(
            label_features,
            padding=self.padding,
            return_tensors="pt",
        )
        
        # replace padding with -100 to ignore loss correctly
        batch["labels"] = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        
        return batch

if __name__ == "__main__":
    TRAIN_EPOCHS = 10
    BATCH_SIZE = 1
    MAX_SAMPLES=-1 #sample Utterances to use for training.  use -1 to get all the data.
    CALC_HOURS=True
    #
    # Define instance of Collator to pad dataset at run time with 
    # for each batch
    # For the most part, its job is to padd the data 
    # the input data is padded with zeros
    # the output labels are padded with -100's to 
    # avoid effecting the CTC scores.
    #
    data_collator = DataCollatorCTCWithPadding(
        tokenizer=asr_pipeline.tokenizer, 
        feature_extractor=asr_pipeline.feature_extractor, 
    )
    
    #
    # Only read in 1600 samples as this is around 2hrs of data that is used to 
    # get the WER.  This has not bee shuffled so it is problematic distribution-wise.
    #
    dsing_train = load_dataset("audiofolder", data_dir=dataset_folder, split=f'train[:{str(MAX_SAMPLES)}]')
    if CALC_HOURS:
        arr_lens = [len(d['array']) for d in dsing_train['audio']]
        print(f"Total Hours of Training Data: {np.sum(arr_lens)/ target_sampling_rate / 3600:.2f}")
            
    dsing_train = dsing_train.to_iterable_dataset()
    dsing_train = dsing_train.with_format('torch')
    # make changes to dataset object to prepare for Wav2Vec2 model
    dsing_train = dsing_train.map(
        prepare_dataset, 
        remove_columns=["audio","transcription"], 
        fn_kwargs={'tokenizer':asr_pipeline.tokenizer, 'feature_extractor':asr_pipeline.feature_extractor}
    )

    dsing_val = load_dataset("audiofolder", data_dir=dataset_folder, split='validation[:16]')
    dsing_val = dsing_val.to_iterable_dataset()
    dsing_val = dsing_val.with_format('torch')
    # make changes to dataset object to prepare for Wav2Vec2 model
    dsing_val = dsing_val.map(
        prepare_dataset, 
        remove_columns=["audio","transcription"], 
        fn_kwargs={'tokenizer':asr_pipeline.tokenizer, 'feature_extractor':asr_pipeline.feature_extractor}
    )

    training_args = TrainingArguments(
        output_dir="test_trainer",
        overwrite_output_dir = True,
        max_steps=8797 / BATCH_SIZE * NUM_EPOCHS,      # ANymore than 5k, steps and I run out of memory. 
                            # Should be MAX_SAMPLES / BATCH_SIZE * Epochs
        # gradient_checkpointing
        # the trade off is  O(sqrt(n)) savings 
        # with implemented memory-wise, at the 
        # cost of performing one additional 
        # forward pass.
        gradient_checkpointing=True,   
        #use_cpu=True,
        fp16=True,                      #use when we are doing the GPU based training
        fp16_full_eval=True,
        #resume_from_checkpoint=True,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        evaluation_strategy="steps",
        eval_steps=500,
        logging_dir='./logs',
        learning_rate=1e-4,              # Based on fairseq yaml
        weight_decay=0.005,
        warmup_steps=1000,
        save_strategy='steps',
        save_steps=500, # this is a ratio of the current step / next.
        metric_for_best_model='loss',
        save_total_limit=2,
        report_to='tensorboard',                 #logging thing.  there was a warning.
        logging_strategy='steps',
        logging_steps=10,
    )

    # Create HuggingFace Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dsing_train,
        eval_dataset=dsing_val,
        compute_metrics=compute_metrics_dummy,
        tokenizer=asr_pipeline.feature_extractor,
    )

    start = time.time()
    trainer.train()
    finish = time.time()
    print(f"Finished in {(finish-start)/60} minutes.") 
    
