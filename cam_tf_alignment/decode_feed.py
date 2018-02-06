"""Binary for decoding from translation models based on tensorflow/models/rnn/translate/translate.py.

Note that this decoder is greedy and very basic. For a better decoder, see http://ucam-smt.github.io/sgnmt/html/tutorial.html
which supports decoding from tensorflow models.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import numpy as np
import datetime
import logging
import pickle

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

if __name__ == "__main__":
  from cam_tf_alignment.train import FLAGS as train_flags # get default model settings from train.py
from cam_tf_alignment.utils import data_utils, model_utils

# Decoder settings
tf.app.flags.DEFINE_string("test_src_idx", "/tmp/in.txt", "An integer-encoded input file")
tf.app.flags.DEFINE_string("test_out_idx", "/tmp/out.txt", "Output file for decoder output")
tf.app.flags.DEFINE_string("atts_out", None, "Output file for attentions")
tf.app.flags.DEFINE_string("trg_idx", None, "Output file for attentions")
tf.app.flags.DEFINE_integer("max_sentences", 0, "The maximum number of sentences to translate (all if set to 0)")
tf.app.flags.DEFINE_boolean("interactive", False, "Decode from command line")
tf.app.flags.DEFINE_boolean("entropy", False, "Add reducing entropy as criteria")
FLAGS = tf.app.flags.FLAGS

def decode(config, input_file=None, output=None, max_sentences=0):
  if input_file and output:
    inp = input_file
    out = output
    out_atts = None
    trg_idx = None
  else:
    inp = config['test_src_idx']
    out = config['test_out_idx']
    out_atts = config['atts_out']
    trg_idx = config['trg_idx']

  # Find longest input/output to create suitable bucket
  max_input_length = 0
  with open(inp) as f_in:
    for sentence in f_in:
      token_ids = [ int(tok) for tok in sentence.strip().split() ]
      if config['add_src_eos']:
        token_ids.append(data_utils.EOS_ID)
      if len(token_ids) > max_input_length:
        max_input_length = len(token_ids)
  max_output_length = 0
  with open(trg_idx) as f_out:
    for sentence in f_out:
      trg_ids = [ int(tok) for tok in sentence.strip().split() ]
      if config['add_src_eos']:
        trg_ids.append(data_utils.EOS_ID)
      if len(trg_ids) > max_output_length:
        max_output_length = len(trg_ids)
  bucket = model_utils.make_bucket(max([max_input_length, max_output_length]), greedy_decoder=True)
  buckets = list(model_utils._buckets)
  buckets.append(bucket)
  logging.info("Add new bucket={}".format(bucket))

  with tf.Session() as session:
    # Create model and load parameters: uses the training graph for decoding
    model = model_utils.create_model(session, config, forward_only="do_decode",
                                        buckets=buckets)
    model.batch_size = 1  # We decode one sentence at a time.

    # Decode input file
    num_sentences = 0
    max_sents = 0
    if 'max_sentences' in config:
      max_sents = config and config['max_sentences']
    if max_sentences > 0:
      max_sents = max_sentences

    if out_atts:
        open(out_atts, 'wb').close()

    logging.info("Start decoding, max_sentences=%i" % max_sents)
    with open(inp) as f_in, open(trg_idx, 'r') as f_trg, open(out, 'w') as f_out:
      for sentence in f_in:
        trg = f_trg.readline()
        outputs, attns = get_outputs(session, config, model, sentence, trg, buckets)
        logging.info("Output: {}".format(outputs))

        # If there is an EOS symbol in outputs, cut them at that point.
        if data_utils.EOS_ID in outputs:
          outputs = outputs[:outputs.index(data_utils.EOS_ID)]
        print(" ".join([str(tok) for tok in outputs]), file=f_out)

        # Dump the attentions to a file
        if out_atts:
            with open(out_atts, "ab") as f:
                pickle.dump(attns, f, pickle.HIGHEST_PROTOCOL)

        num_sentences += 1
        if max_sents > 0 and num_sentences >= max_sents:
          break
      logging.info("Decoding completed.")

def decode_interactive(config):
  with tf.Session() as session:
    # Create model and load parameters: uses the training graph for decoding
    model = model_utils.create_model(session, config, forward_only=True)
    model.batch_size = 1  # We decode one sentence at a time.

    # Decode from standard input.
    sys.stdout.write("> ")
    sys.stdout.flush()
    sentence = sys.stdin.readline()
    while sentence:
      outputs = get_outputs(session, config, model, sentence)
      print("Output: %s" % " ".join([str(tok) for tok in outputs]))
      print("> ", end="")
      sys.stdout.flush()
      sentence = sys.stdin.readline()

def get_outputs(session, config, model, sentence, trg, buckets=None):
  # Get token-ids for the input sentence.
  token_ids = [ int(tok) for tok in sentence.strip().split() ]
  token_ids = [ w if w < config['src_vocab_size'] else data_utils.UNK_ID
                for w in token_ids ]
  if config['add_src_eos']:
    token_ids.append(data_utils.EOS_ID)

  trg_ids = [ int(tok) for tok in trg.strip().split() ]
  trg_ids = [ w if w < config['trg_vocab_size'] else data_utils.UNK_ID
                for w in trg_ids ]

  if not buckets:
    buckets = model_utils._buckets
  bucket_id = min([b for b in xrange(len(buckets))
                  if buckets[b][0] >= len(token_ids) and buckets[b][1] >= len(trg_ids)])
  logging.info("Bucket {}".format(buckets[bucket_id]))
  logging.info("Input: {}".format(token_ids))
  logging.info("Output: {}".format(trg_ids))

  # Get a 1-element batch to feed the sentence to the model.
  encoder_inputs, decoder_inputs, target_weights, sequence_length, src_mask, bow_mask = model.get_batch(
    {bucket_id: [(token_ids, trg_ids)]}, bucket_id, config['encoder'])

  # Get output logits for the sentence.
  _, _, output_logits, attns = model.step(session, encoder_inputs, decoder_inputs,
                                 target_weights, bucket_id, forward_only="do_decode",
                                 sequence_length=sequence_length, src_mask=src_mask, bow_mask=bow_mask)

  # This is a greedy decoder - outputs are just argmaxes of output_logits.
  outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]
  return outputs, attns

def main(_):
  config = model_utils.process_args(FLAGS, train=False, greedy_decoder=True, special_decode=True)
  if FLAGS.interactive:
    decode_interactive(config)
  else:
    decode(config)

if __name__ == "__main__":
  logging.getLogger().setLevel(logging.INFO)
  logging.info("Start: {}".format(datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S')))
  tf.app.run()
  logging.info("End: {}".format(datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S')))

