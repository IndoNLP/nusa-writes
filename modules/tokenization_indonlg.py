# coding=utf-8
# Copyright 2018 Google AI, Google Brain and Carnegie Mellon University Authors and the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License
""" Tokenization classes for IndoNLG model."""

import os
from shutil import copyfile
from typing import TYPE_CHECKING, Any, Dict, List, NamedTuple, Optional, Sequence, Tuple, Union
from transformers import PreTrainedTokenizer, BatchEncoding

from collections.abc import Mapping
from transformers.utils import (
    EntryNotFoundError,
    ExplicitEnum,
    PaddingStrategy,
    PushToHubMixin,
    RepositoryNotFoundError,
    RevisionNotFoundError,
    TensorType,
    add_end_docstrings,
    cached_path,
    copy_func,
    get_file_from_repo,
    hf_bucket_url,
    is_flax_available,
    is_offline_mode,
    is_remote_url,
    is_tf_available,
    is_tokenizers_available,
    is_torch_available,
    logging,
    to_py_obj,
    torch_required,
)
import sentencepiece as spm

from transformers.utils import logging
from transformers.utils.generic import _is_jax, _is_numpy, _is_tensorflow, _is_torch, _is_torch_device

logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "sentencepiece.bpe.model"}

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "indobenchmark/indobart": "https://huggingface.co/indobenchmark/indobart/resolve/main/sentencepiece.bpe.model",
        "indobenchmark/indogpt": "https://huggingface.co/indobenchmark/indogpt/resolve/main/sentencepiece.bpe.model",
        "indobenchmark/indobart-v2": "https://huggingface.co/indobenchmark/indobart-v2/resolve/main/sentencepiece.bpe.model"
    }
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "indobenchmark/indobart": 768,
    "indobenchmark/indogpt": 768,
    "indobenchmark/indobart-v2": 768
}

SHARED_MODEL_IDENTIFIERS = [
    # Load with
    "indobenchmark/indobart",
    "indobenchmark/indogpt",
    "indobenchmark/indobart-v2"
]

SPIECE_UNDERLINE = "▁"

# Define type aliases and NamedTuples
TextInput = str
PreTokenizedInput = List[str]
EncodedInput = List[int]
TextInputPair = Tuple[str, str]
PreTokenizedInputPair = Tuple[List[str], List[str]]
EncodedInputPair = Tuple[List[int], List[int]]

class IndoNLGTokenizer(PreTrainedTokenizer):
    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    model_input_names=['input_ids', 'attention_mask', 'decoder_input_ids', 'decoder_attention_mask', 'labels']
    input_error_message = "text input must of type `str` (single example), `List[str]` (batch of examples)."

    def __init__(
        self,
        vocab_file,
        decode_special_token=True,
        bos_token="<s>",
        eos_token="</s>",
        sep_token="</s>",
        cls_token="<s>",
        unk_token="<unk>",
        pad_token="<pad>",
        mask_token="<mask>",
        additional_special_tokens=[],
        **kwargs
    ):
        super().__init__(
            vocab_file=vocab_file,
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            sep_token=sep_token,
            cls_token=cls_token,
            pad_token=pad_token,
            mask_token=mask_token,
            additional_special_tokens=additional_special_tokens,
            **kwargs,
        )
        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.Load(str(vocab_file))
        self.vocab_file = vocab_file
        self.decode_special_token = decode_special_token
        self.model_max_length = 1024
        
        # HACK: These tokens were added by fairseq but don't seem to be actually used when duplicated in the actual
        # sentencepiece vocabulary (this is the case for <s> and </s>
        self.special_tokens_to_ids = {
            "[javanese]": 40000, 
            "[sundanese]": 40001, 
            "[indonesian]": 40002,
            "<mask>": 40003
        }
        self.special_ids_to_tokens = {v: k for k, v in self.special_tokens_to_ids.items()}
        
        # Store Language token ID
        self.javanese_token = '[javanese]'
        self.javanese_token_id = 40000
        self.sundanese_token = '[sundanese]'
        self.sundanese_token_id = 40001
        self.indonesian_token = '[indonesian]'
        self.indonesian_token_id = 40002
        
        self.special_token_ids = [
            self.bos_token_id, self.eos_token_id, self.sep_token_id, self.cls_token_id, 
            self.unk_token_id, self.pad_token_id, self.mask_token_id,
            self.javanese_token_id, self.sundanese_token_id, self.indonesian_token_id
        ]
    
    def prepare_input_for_generation(self, inputs, model_type='indobart', lang_token='[indonesian]', decoder_inputs=None,
                                             decoder_lang_token='[indonesian]', padding='longest', return_tensors=None):
        """
        Build model inputs for a specified `model_type`. There are two possible `model_type`, i.e., indobart and indogpt.
        
        When `model_type` is indogpt, `lang_token`, `decoder_inputs`, and `decoder_lang_token` parameters will be ignored 
        and the input will be encoded in the gpt2 sequence format as follow: 
        
        - indogpt sequence: ``<s> X``
        
        When `model_type` is indobart, `inputs` and `lang_token` are used as the sequence and language identifier for the indobart encoder, 
        while `decoder_inputs` and `decoder_lang_token` are used as the sequence and language identifier of the decoder
        
        - indobart encoder sequence: ``X </s> <lang_token_id>``
        - indobart decoder sequences: ``<decoder_lang_token_id> X </s>``

        Args:
            inputs (:obj:`str` or `List[str]`):
                text sequence or list of text sequences to be tokenized.
            model_type (:obj:`str`, defaults to :obj:`indobart`):
                model type to determine the format of the tokenized sequence. Valid values are `indobart` and `indogpt`.
            lang_token (:obj:`str`, defaults to :obj:`[indonesian]`):
                language token to determine the format of the tokenized sequence. Valid values are `[indonesian]`, `[sundanese], and [javanese]`.
            decoder_inputs (:obj:`str` or `List[str]`, `optional`):
                decoder text sequence or list of text sequences to be tokenized.
            decoder_lang_token (:obj:`str`, defaults to :obj:`[indonesian]`):
                decoder language token to determine the format of the tokenized sequence. Valid values are `[indonesian]`, `[sundanese], and [javanese]`.
            padding (:obj:`str`, defaults to :obj:`longest`):
                padding strategy to pad the tokenized sequences. Valid values are `longest`, `max_length`, and `do_not_pad`.
            return_tensors (:obj:`str`, defaults to :obj:`None`):
                Returned tensor type of the tokenized sequence. When set to `None`, the return type will be List[int]. Valid values are `None`, `pt`, and `tf`

        Returns:
            :obj:`Dict`: Dictionary with `input_ids`, `attention_mask`, `decoder_input_ids` (optional), and `decoder_attention_mask` (optional)
        """        
        if model_type == 'indogpt':
            # Process indogpt input
            if type(inputs) == str:
                 return self(f'<s> {inputs}', padding=padding, return_tensors=return_tensors)
            elif type(inputs) == list:
                if len(inputs) == 0 or type(inputs[0]) != str:
                    raise ValueError(IndoNLGTokenizer.input_error_message)
                else:
                    return self([f'<s> {input_data}' for input_data in inputs], padding=padding, return_tensors=return_tensors)
            else:
                raise ValueError(IndoNLGTokenizer.input_error_message)
        elif model_type == 'indobart':
                                     
            # Process encoder input
            if lang_token not in self.special_tokens_to_ids:
                raise ValueError(f"Unknown lang_token `{lang_token}`, lang_token must be either `[javanese]`, `[sundanese]`, or `[indonesian]`")  
            elif type(inputs) == list:
                if len(inputs) == 0 or type(inputs[0]) != str:
                    raise ValueError(IndoNLGTokenizer.input_error_message)
            elif type(inputs) != str:
                raise ValueError(IndoNLGTokenizer.input_error_message)
                
            lang_id = self.special_tokens_to_ids[lang_token]
            input_batch = self(inputs, return_attention_mask=False)
            if type(inputs) == str:
                input_batch['input_ids'] = [self.bos_token_id] + input_batch['input_ids'] + [self.eos_token_id, lang_id]
            else:
                input_batch['input_ids'] = list(map(lambda input_ids: [self.bos_token_id] + input_ids + [self.eos_token_id, lang_id], input_batch['input_ids']))
            
            if decoder_inputs is None:
                # Return encoder input
                return self.pad(input_batch, return_tensors=return_tensors)
            else:
                # Process decoder input
                if decoder_lang_token not in self.special_tokens_to_ids:
                    raise ValueError(f"Unknown decoder_lang_token `{decoder_lang_token}`, decoder_lang_token must be either `[javanese]`, `[sundanese]`, or `[indonesian]`")  
                elif type(decoder_inputs) == list:
                    if len(decoder_inputs) == 0:
                        raise ValueError(IndoNLGTokenizer.input_error_message)
                    elif type(decoder_inputs[0]) != str:
                        raise ValueError(IndoNLGTokenizer.input_error_message)
                elif type(decoder_inputs) != str:
                    raise ValueError(IndoNLGTokenizer.input_error_message)

                decoder_lang_id = self.special_tokens_to_ids[decoder_lang_token]
                decoder_input_batch = self(decoder_inputs, return_attention_mask=False)
                
                if type(decoder_inputs) == str:
                    labels = [self.bos_token_id] + decoder_input_batch['input_ids'] + [self.eos_token_id, decoder_lang_id]
                    decoder_input_batch['input_ids'] = [decoder_lang_id, self.bos_token_id] + decoder_input_batch['input_ids'] + [self.eos_token_id]
                else:
                    labels = list(map(lambda input_ids: [self.bos_token_id] + input_ids + [self.eos_token_id, decoder_lang_id], decoder_input_batch['input_ids']))
                    decoder_input_batch['input_ids'] = list(map(lambda input_ids: [decoder_lang_id, self.bos_token_id] + input_ids + [self.eos_token_id], decoder_input_batch['input_ids']))
                    
                # Padding
                input_batch = self.pad(input_batch, return_tensors=return_tensors)
                decoder_input_batch = self.pad(decoder_input_batch, return_tensors=return_tensors)
                labels = self.pad({'input_ids': labels}, return_tensors=return_tensors)['input_ids']
                if not isinstance(labels, (list, tuple)):
                    labels[labels == self.pad_token_id] = -100
                else:
                    labels = list(map(lambda x: -100 if x == self.pad_token_id else x, labels))
                
                # Store into a single dict
                input_batch['decoder_input_ids'] = decoder_input_batch['input_ids']
                input_batch['decoder_attention_mask'] = decoder_input_batch['attention_mask']
                input_batch['labels'] = labels
                
                return input_batch

    def __len__(self):
        return max(self.special_ids_to_tokens) + 1
    
    def get_special_tokens_mask(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, already_has_special_tokens: bool = False
    ) -> List[int]:
        """
        Retrieve sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer ``prepare_for_model`` method.

        Args:
            token_ids_0 (:obj:`List[int]`):
                List of IDs.
            token_ids_1 (:obj:`List[int]`, `optional`):
                Optional second list of IDs for sequence pairs.
            already_has_special_tokens (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not the token list is already formatted with special tokens for the model.

        Returns:
            :obj:`List[int]`: A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
        """
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )

        if token_ids_1 is None:
            return [1] + ([0] * len(token_ids_0)) + [1]
        return [1] + ([0] * len(token_ids_0)) + [1, 1] + ([0] * len(token_ids_1)) + [1]

    @property
    def vocab_size(self):
        return 4 + len(self.sp_model)

    def get_vocab(self):
        vocab = {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size)}
        vocab.update(self.added_tokens_encoder)
        return vocab

    def _tokenize(self, text: str) -> List[str]:
        return self.sp_model.encode(text.lower(), out_type=str)
    
    def convert_ids_to_tokens(
        self, ids: Union[int, List[int]], skip_special_tokens: bool = False
    ) -> Union[str, List[str]]:
        """
        Converts a single index or a sequence of indices in a token or a sequence of tokens, using the vocabulary and
        added tokens.
        Args:
            ids (`int` or `List[int]`):
                The token id (or token ids) to convert to tokens.
            skip_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not to remove special tokens in the decoding.
        Returns:
            `str` or `List[str]`: The decoded token(s).
        """
        if isinstance(ids, int):
            if ids not in self.added_tokens_decoder or ids in self.special_tokens_to_ids:
                return self._convert_id_to_token(ids, skip_special_tokens=skip_special_tokens)
            else:
                return self.added_tokens_decoder[ids]
        tokens = []
        for index in ids:
            index = int(index)
            if skip_special_tokens and index in (self.all_special_ids + list(self.special_tokens_to_ids.values())):
                continue
            if index not in self.added_tokens_decoder or index in self.special_tokens_to_ids:
                tokens.append(self._convert_id_to_token(index, skip_special_tokens=skip_special_tokens))                
            else:
                tokens.append(self.added_tokens_decoder[index])
        return tokens
    
    def _convert_token_to_id(self, token):
        """ Converts a token (str) in an id using the vocab. """
        if token in self.special_tokens_to_ids:
            return self.special_tokens_to_ids[token]
        return self.sp_model.PieceToId(token)
    
    def _convert_id_to_token(self, index, skip_special_tokens=False):
        """Converts an index (integer) in a token (str) using the vocab."""
        if skip_special_tokens and index in self.special_token_ids:
            return ''
            
        if index in self.special_ids_to_tokens:
            return self.special_ids_to_tokens[index]
        
        token = self.sp_model.IdToPiece(index)
        if '<0x' in token:
            char_rep = chr(int(token[1:-1], 0))
            if char_rep.isprintable():
                return char_rep
        return token
    
    def __getstate__(self):
        state = self.__dict__.copy()
        state["sp_model"] = None
        return state

    def __setstate__(self, d):
        self.__dict__ = d

        # for backward compatibility
        if not hasattr(self, "sp_model_kwargs"):
            self.sp_model_kwargs = {}

        self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        self.sp_model.Load(self.vocab_file)

    def decode(self, inputs, skip_special_tokens=False):     
        outputs = super().decode(inputs, skip_special_tokens=skip_special_tokens)
        return outputs.replace(' ','').replace('▁', ' ')
    
    def _pad_decoder(
        self,
        encoded_inputs: Union[Dict[str, EncodedInput], BatchEncoding],
        max_length: Optional[int] = None,
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
        pad_to_multiple_of: Optional[int] = None,
        return_attention_mask: Optional[bool] = None,
    ) -> dict:
        """
        Pad encoded inputs (on left/right and up to predefined length or max length in the batch)
        Args:
            encoded_inputs:
                Dictionary of tokenized inputs (`List[int]`) or batch of tokenized inputs (`List[List[int]]`).
            max_length: maximum length of the returned list and optionally padding length (see below).
                Will truncate by taking into account the special tokens.
            padding_strategy: PaddingStrategy to use for padding.
                - PaddingStrategy.LONGEST Pad to the longest sequence in the batch
                - PaddingStrategy.MAX_LENGTH: Pad to the max length (default)
                - PaddingStrategy.DO_NOT_PAD: Do not pad
                The tokenizer padding sides are defined in self.padding_side:
                    - 'left': pads on the left of the sequences
                    - 'right': pads on the right of the sequences
            pad_to_multiple_of: (optional) Integer if set will pad the sequence to a multiple of the provided value.
                This is especially useful to enable the use of Tensor Core on NVIDIA hardware with compute capability
                >= 7.5 (Volta).
            return_attention_mask:
                (optional) Set to False to avoid returning attention mask (default: set to model specifics)
        """
        # Load from model defaults
        if return_attention_mask is None:
            return_attention_mask = "decoder_attention_mask" in self.model_input_names

        required_input = encoded_inputs[self.model_input_names[2]]

        if max_length is not None and pad_to_multiple_of is not None and (max_length % pad_to_multiple_of != 0):
            max_length = ((max_length // pad_to_multiple_of) + 1) * pad_to_multiple_of

        needs_to_be_padded = padding_strategy != PaddingStrategy.DO_NOT_PAD and len(required_input) != max_length

        # Initialize attention mask if not present.
        if return_attention_mask and "decoder_attention_mask" not in encoded_inputs:
            encoded_inputs["decoder_attention_mask"] = [1] * len(required_input)

        if needs_to_be_padded:
            difference = max_length - len(required_input)

            if self.padding_side == "right":
                if return_attention_mask:
                    encoded_inputs["decoder_attention_mask"] = encoded_inputs["decoder_attention_mask"] + [0] * difference
                if "decoder_token_type_ids" in encoded_inputs:
                    encoded_inputs["decoder_token_type_ids"] = (
                        encoded_inputs["decoder_token_type_ids"] + [self.pad_token_type_id] * difference
                    )
                if "decoder_special_tokens_mask" in encoded_inputs:
                    encoded_inputs["decoder_special_tokens_mask"] = encoded_inputs["decoder_special_tokens_mask"] + [1] * difference
                encoded_inputs[self.model_input_names[2]] = required_input + [self.pad_token_id] * difference
                
                label_input = encoded_inputs[self.model_input_names[4]]
                encoded_inputs[self.model_input_names[4]] = label_input + [-100] * difference
            elif self.padding_side == "left":
                if return_attention_mask:
                    encoded_inputs["decoder_attention_mask"] = [0] * difference + encoded_inputs["decoder_attention_mask"]
                if "decoder_token_type_ids" in encoded_inputs:
                    encoded_inputs["decoder_token_type_ids"] = [self.pad_token_type_id] * difference + encoded_inputs[
                        "decoder_token_type_ids"
                    ]
                if "decoder_special_tokens_mask" in encoded_inputs:
                    encoded_inputs["decoder_special_tokens_mask"] = [1] * difference + encoded_inputs["decoder_special_tokens_mask"]
                encoded_inputs[self.model_input_names[2]] = [self.pad_token_id] * difference + required_input
                
                label_input = encoded_inputs[self.model_input_names[4]]
                encoded_inputs[self.model_input_names[4]] = label_input + [-100] * difference
            else:
                raise ValueError("Invalid padding strategy:" + str(self.padding_side))

        return encoded_inputs    
        
    def pad(self,
        encoded_inputs: Union[
            BatchEncoding,
            List[BatchEncoding],
            Dict[str, EncodedInput],
            Dict[str, List[EncodedInput]],
            List[Dict[str, EncodedInput]],
        ],
        padding: Union[bool, str, PaddingStrategy] = True,
        max_length: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
        return_attention_mask: Optional[bool] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        verbose: bool = True,
    ) -> BatchEncoding:
        """
        Pad a single encoded input or a batch of encoded inputs up to predefined length or to the max sequence length
        in the batch.

        Padding side (left/right) padding token ids are defined at the tokenizer level (with `self.padding_side`,
        `self.pad_token_id` and `self.pad_token_type_id`)

        <Tip>

        If the `encoded_inputs` passed are dictionary of numpy arrays, PyTorch tensors or TensorFlow tensors, the
        result will use the same type unless you provide a different tensor type with `return_tensors`. In the case of
        PyTorch tensors, you will lose the specific device of your tensors however.

        </Tip>

        Args:
            encoded_inputs ([`BatchEncoding`], list of [`BatchEncoding`], `Dict[str, List[int]]`, `Dict[str, List[List[int]]` or `List[Dict[str, List[int]]]`):
                Tokenized inputs. Can represent one input ([`BatchEncoding`] or `Dict[str, List[int]]`) or a batch of
                tokenized inputs (list of [`BatchEncoding`], *Dict[str, List[List[int]]]* or *List[Dict[str,
                List[int]]]*) so you can use this method during preprocessing as well as in a PyTorch Dataloader
                collate function.

                Instead of `List[int]` you can have tensors (numpy arrays, PyTorch tensors or TensorFlow tensors), see
                the note above for the return type.
            padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
                 Select a strategy to pad the returned sequences (according to the model's padding side and padding
                 index) among:

                - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
                  sequence if provided).
                - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
                  acceptable input length for the model if that argument is not provided.
                - `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different
                  lengths).
            max_length (`int`, *optional*):
                Maximum length of the returned list and optionally padding length (see above).
            pad_to_multiple_of (`int`, *optional*):
                If set will pad the sequence to a multiple of the provided value.

                This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability
                >= 7.5 (Volta).
            return_attention_mask (`bool`, *optional*):
                Whether to return the attention mask. If left to the default, will return the attention mask according
                to the specific tokenizer's default, defined by the `return_outputs` attribute.

                [What are attention masks?](../glossary#attention-mask)
            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                If set, will return tensors instead of list of python integers. Acceptable values are:

                - `'tf'`: Return TensorFlow `tf.constant` objects.
                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return Numpy `np.ndarray` objects.
            verbose (`bool`, *optional*, defaults to `True`):
                Whether or not to print more information and warnings.
        """
        # If we have a list of dicts, let's convert it in a dict of lists
        # We do this to allow using this method as a collate_fn function in PyTorch Dataloader
        if isinstance(encoded_inputs, (list, tuple)) and isinstance(encoded_inputs[0], Mapping):
            encoded_inputs = {key: [example[key] for example in encoded_inputs] for key in encoded_inputs[0].keys()}

        # The model's main input name, usually `input_ids`, has be passed for padding
        if self.model_input_names[0] not in encoded_inputs:
            raise ValueError(
                "You should supply an encoding or a list of encodings to this method "
                f"that includes {self.model_input_names[0]}, but you provided {list(encoded_inputs.keys())}"
            )

        required_input = encoded_inputs[self.model_input_names[0]]

        if not required_input:
            if return_attention_mask:
                encoded_inputs["attention_mask"] = []
            return encoded_inputs

        # If we have PyTorch/TF/NumPy tensors/arrays as inputs, we cast them as python objects
        # and rebuild them afterwards if no return_tensors is specified
        # Note that we lose the specific device the tensor may be on for PyTorch

        first_element = required_input[0]
        if isinstance(first_element, (list, tuple)):
            # first_element might be an empty list/tuple in some edge cases so we grab the first non empty element.
            for item in required_input:
                if len(item) != 0:
                    first_element = item[0]
                    break
        # At this state, if `first_element` is still a list/tuple, it's an empty one so there is nothing to do.
        if not isinstance(first_element, (int, list, tuple)):
            if is_tf_available() and _is_tensorflow(first_element):
                return_tensors = "tf" if return_tensors is None else return_tensors
            elif is_torch_available() and _is_torch(first_element):
                return_tensors = "pt" if return_tensors is None else return_tensors
            elif isinstance(first_element, np.ndarray):
                return_tensors = "np" if return_tensors is None else return_tensors
            else:
                raise ValueError(
                    f"type of {first_element} unknown: {type(first_element)}. "
                    f"Should be one of a python, numpy, pytorch or tensorflow object."
                )

            for key, value in encoded_inputs.items():
                encoded_inputs[key] = to_py_obj(value)

        # Convert padding_strategy in PaddingStrategy
        padding_strategy, _, max_length, _ = self._get_padding_truncation_strategies(
            padding=padding, max_length=max_length, verbose=verbose
        )

        required_input = encoded_inputs[self.model_input_names[0]]
        if required_input and not isinstance(required_input[0], (list, tuple)):
            encoded_inputs = self._pad(
                encoded_inputs,
                max_length=max_length,
                padding_strategy=padding_strategy,
                pad_to_multiple_of=pad_to_multiple_of,
                return_attention_mask=return_attention_mask,
            )
            return BatchEncoding(encoded_inputs, tensor_type=return_tensors)

        batch_size = len(required_input)
        assert all(
            len(v) == batch_size for v in encoded_inputs.values()
        ), "Some items in the output dictionary have a different batch size than others."

        if padding_strategy == PaddingStrategy.LONGEST:
            max_length = max(len(inputs) for inputs in required_input)
            padding_strategy = PaddingStrategy.MAX_LENGTH

        batch_outputs = {}
        for i in range(batch_size):
            inputs = dict((k, v[i]) for k, v in encoded_inputs.items())
            outputs = self._pad(
                inputs,
                max_length=max_length,
                padding_strategy=padding_strategy,
                pad_to_multiple_of=pad_to_multiple_of,
                return_attention_mask=return_attention_mask,
            )
            
            # Handle decoder_input_ids
            if self.model_input_names[2] in outputs:
                max_decoder_length = max(len(inputs) for inputs in encoded_inputs[self.model_input_names[2]])                    
                outputs = self._pad_decoder(
                    outputs,
                    max_length=max_decoder_length,
                    padding_strategy=padding_strategy,
                    pad_to_multiple_of=pad_to_multiple_of,
                    return_attention_mask=return_attention_mask,
                )

            for key, value in outputs.items():
                if key not in batch_outputs:
                    batch_outputs[key] = []
                batch_outputs[key].append(value)

        return BatchEncoding(batch_outputs, tensor_type=return_tensors)