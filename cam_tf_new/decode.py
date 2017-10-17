"""Binary for decoding from translation models based on tensorflow/models/rnn/translate/translate.py.
Note that this decoder is greedy and very basic. For a better decoder, see http://ucam-smt.github.io/sgnmt/html/tutorial.html
which supports decoding from tensorflow models.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cPickle
import sys
import numpy as np
import datetime
import logging
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

if __name__ == "__main__":
  from cam_tf_new.train import FLAGS as train_flags # get default model settings from train.py
from cam_tf_new.utils import data_utils, model_utils

# Decoder settings
tf.app.flags.DEFINE_string("test_src_idx", "/tmp/in.txt", "An integer-encoded input file")
tf.app.flags.DEFINE_string("test_out_idx", "/tmp/out.txt", "Output file for decoder output")
tf.app.flags.DEFINE_string("output_hidden", "/tmp/hidden", "Output file for hidden state")
tf.app.flags.DEFINE_integer("max_sentences", 0, "The maximum number of sentences to translate (all if set to 0)")
tf.app.flags.DEFINE_string("decode_hidden", None, "Decode from hidden layers in file")
tf.app.flags.DEFINE_boolean("interactive", False, "Decode from command line")
tf.app.flags.DEFINE_string("decode_interpolate_hidden", None, "Decode from hidden layers interpolating between true and generated hidden vectors")
FLAGS = tf.app.flags.FLAGS

def output_increment_sentence(outputs, f_out, num_sentences):
  logging.info("Output: {}".format(outputs))
  # If there is an EOS symbol in outputs, cut them at that point.
  if data_utils.EOS_ID in outputs:
    outputs = outputs[:outputs.index(data_utils.EOS_ID)]
  print(" ".join([str(tok) for tok in outputs]), file=f_out)
  return num_sentences + 1

def get_max_inp_len(input_file, add_src_eos):
  max_input_length = 0
  eos_token_len = 1 if add_src_eos else 0
  with open(input_file) as f_in:
    for sentence in f_in:
      sentence_len = len(sentence.strip().split()) + eos_token_len
      max_input_length = max(sentence_len, max_input_length)
  return max_input_length

def get_model_buckets(config, input_file):
  max_input_length = get_max_inp_len(input_file, config['add_src_eos'])
  buckets = list(model_utils._buckets)
  logging.info("Decoder buckets: {}".format(buckets))
  max_bucket = buckets[len(buckets) - 1][0]
  if max_input_length > max_bucket:
    if config['grammar_def']:
      max_len = max(2 * max_input_length, config['max_target_length'] + 50)
      bucket = model_utils.make_bucket(max_input_length, greedy_decoder=True, max_trg_len=max_len)
    else:
      bucket = model_utils.make_bucket(max_input_length, greedy_decoder=True,
                                       max_trg_len=config['max_target_length'] )
    buckets.append(bucket)
    logging.info("Add new bucket={}".format(bucket))
  return buckets

def get_inference_model(config, session, buckets=None, hidden=False):
  model = model_utils.create_model(session, config, forward_only=True, buckets=buckets, hidden=hidden)
  model.batch_size = 1
  return model


class TokParsePredictor(object):
  def __init__(self, grammar_path, word_out=False):
    logging.info('Initialising decoder grammar')
    self.grammar_path = grammar_path
    self.word_out = word_out
    self.stack = [data_utils.EOS_ID, data_utils.GO_ID]
    self.current_lhs = None
    self.current_rhs = []
    self.keep_lhs = False
    self.prepare_grammar()

  def prepare_grammar(self):
    self.lhs_to_can_follow = {}
    with open(self.grammar_path) as f:
      for line in f:
        nt, rule = line.split(':')
        nt = int(nt.strip())
        self.lhs_to_can_follow[nt] = set([int(r) for r in rule.strip().split()])
      self.last_nt_in_rule = {nt: True for nt in self.lhs_to_can_follow}
      for nt, following in self.lhs_to_can_follow.items():
        if 0 in following:
          following.remove(0)
          self.last_nt_in_rule[nt] = False
                    
  def is_nt(self, word):
    if word in self.lhs_to_can_follow:
      return True
    return False

  def reset(self):
    self.stack = [data_utils.EOS_ID, data_utils.GO_ID]
    self.current_lhs = None
    self.keep_lhs = False
    self.current_rhs = []

  def predict_next(self, nmt_posterior, predicting_next_word=False):
    if not self.keep_lhs:
      self.stack.extend(reversed(self.current_rhs))
      self.current_lhs = self.stack.pop()
      self.current_rhs = []
    outgoing_rules = self.lhs_to_can_follow[self.current_lhs]
    for rule_id in range(len(nmt_posterior[0])):
      if rule_id not in outgoing_rules:
        nmt_posterior[0][rule_id] = -np.inf
      
  def consume(self, word):
    if self.is_nt(word):
      self.current_rhs.append(word)
      self.keep_lhs = not self.last_nt_in_rule[word]
    else:
      self.keep_lhs = False

def single_step_decoding(config, input_file, output_file, max_sents=0):
  logging.info('Single-step decoding for grammar')
  max_seq_len = get_max_inp_len(input_file, config['add_src_eos'])
  num_heads = 1
  num_sentences = 0
  grammar = None
  if config['grammar_def']:
    grammar = TokParsePredictor(config['grammar_def'])
  with tf.Session() as session:
    model, train_graph, enc_graph, single_step_dec_graph, buckets = model_utils.load_model(session, config, max_seq_len=max_seq_len)
    model.batch_size = 1
    with open(input_file) as f_in, open(output_file, 'w') as f_out:
      for sentence in f_in:
        dec_in = [data_utils.GO_ID]
        dec_state = {}
        word_count = 0
        token_ids = get_token_ids(sentence, config)
        bucket_id = get_src_bucket_id(token_ids, buckets=buckets)
        enc_in, _, _, seq_len, src_mask, _ = train_graph.get_batch(
          {bucket_id: [(token_ids, [])]}, bucket_id, config['encoder'])
        dec_state['dec_state'], enc_out = enc_graph.encode(session, enc_in, bucket_id, seq_len)
        for a in xrange(num_heads):
          dec_state["dec_attns_%d" % a] = np.zeros((1, enc_out['enc_v_0'].size), dtype=np.float32)
        if config['use_src_mask']:
          dec_state["src_mask"] = src_mask
        output_tokens = []
        while True:
          logit, dec_state = single_step_dec_graph.decode(session, enc_out, dec_state,
                                                          dec_in, bucket_id, config['use_src_mask'],
                                                          word_count, config['use_bow_mask'])
          if grammar:
            grammar.predict_next(logit)
          word_count += 1
          output_tokens.append(int(np.argmax(logit, axis=1)))
          dec_in = [output_tokens[-1]]
          if grammar:
            grammar.consume(dec_in[0])
          if dec_in[0] == data_utils.EOS_ID:
            break
          if word_count > int(config['max_len_factor'] * len(token_ids)):
            logging.info('Incomplete hypothesis')
            break
        if grammar:
          grammar.reset()
        num_sentences = output_increment_sentence(output_tokens, f_out, num_sentences)
        if max_sents > 0 and num_sentences >= max_sents:
          break

def decode(config, input_file=None, output=None, max_sentences=0):
  if input_file and output:
    inp = input_file
    out = output
  else:
    inp = config['test_src_idx']
    out = config['test_out_idx']
  hidden = None
  if 'output_hidden' in config:
    hidden = config['output_hidden']
  max_sents = 0
  if 'max_sentences' in config:
    max_sents = config and config['max_sentences']
  if max_sentences > 0:
    max_sents = max_sentences

  if 'decode_hidden' in config and config['decode_hidden']:
    unpickle_hidden(config, out, max_sentences=max_sents)
  elif 'decode_interpolate_hidden' in config and config['decode_interpolate_hidden']:
    decode_interpolate_hidden(config, out, max_sentences=max_sents)
  elif config['max_len_factor'] > 0 and config['grammar_def']: # single step decoding
    single_step_decoding(config, inp, out, max_sents)
  else:
    with tf.Session() as session:
      # Create model and load parameters: uses the training graph for decoding
      buckets = get_model_buckets(config, inp)
      model = get_inference_model(config, session, buckets=buckets)
      num_sentences = 0
      logging.info("Start decoding, max_sentences=%i" % max_sents)
      with open(inp) as f_in, open(out, 'w') as f_out:
        if hidden:
          with open(hidden, 'wb') as f_hidden:
            pickler = cPickle.Pickler(f_hidden)
            for sentence in f_in:
              outputs, states = get_outputs(session, config, model, sentence, buckets)
              num_sentences = output_increment_sentence(outputs, f_out, num_sentences)
              pickler.dump({'states': states, 'length': len(outputs)})
              if max_sents > 0 and num_sentences >= max_sents:
                break
        else:
          for sentence in f_in:
            outputs, _ = get_outputs(session, config, model, sentence, buckets)
            num_sentences = output_increment_sentence(outputs, f_out, num_sentences)
            if max_sents > 0 and num_sentences >= max_sents:
              break
  logging.info("Decoding completed.")

def get_resize_dim(config):
  # interpolated and reloaded hidden states may need to be reshaped depending on config
  if config['seq2seq_mode'] == 'vae':
    resize_dim = config['latent_size']
  else:
    resize_dim = config['hidden_size']
    if config['use_lstm']:
      resize_dim *= 2
  return resize_dim

def unpickle_hidden(config, out, max_sentences=0):
  hidden_list = []
  with open(config['decode_hidden'], 'rb') as f_in:
    unpickler = cPickle.Unpickler(f_in)
    while True and (max_sentences == 0 or len(hidden_list) < max_sentences):
      try:
        hidden_list.append(np.array(unpickler.load()['states']))
      except (EOFError):
        break
  with tf.Session() as session:
    model = get_inference_model(config, session, hidden=True)
    decode_hidden(session, model, config, out, hidden_list)


def decode_interpolate_hidden(config, out, max_sentences=0):
  hidden_list = []
  num_decoded = 0
  with tf.Session() as session:
    model = get_inference_model(config, session, hidden=True)
    with open(config['decode_interpolate_hidden'], 'rb') as f_in:
      label_samples = cPickle.load(f_in)
      for label in label_samples:
        log_msg(config['test_out_idx'], 'Gaussian mixture component: {}\n'.format(label))
        for interp_list in label_samples[label]:
          log_msg(config['test_out_idx'], 'New interpolation set\n')
          decode_hidden(session, model, config, out, interp_list, append=True)
          num_decoded += 1
          if num_decoded > max_sentences and max_sentences > 0:
            break
     
def log_msg(f_name, msg):
  with open(f_name, 'a') as f_out:
    f_out.write(msg)

def decode_hidden(session, model, config, out, hidden_list, append=False):
  mode = 'w'
  if append:
    mode = 'a'   
  resize_dim = get_resize_dim(config)
  with open(config['test_out_idx'], mode) as f_out:
    for hidden in hidden_list:
      hidden = hidden.reshape(1, resize_dim)
      outputs, states = get_outputs(session, config, model, sentence='', hidden=hidden)
      output_increment_sentence(outputs, f_out, num_sentences=0)

def decode_interactive(config):
  with tf.Session() as session:
    model = get_inference_model(config, session)
    sys.stdout.write("> ")
    sys.stdout.flush()
    sentence = sys.stdin.readline()
    while sentence:
      outputs, states = get_outputs(session, config, model, sentence)
      print("Output: %s" % " ".join([str(tok) for tok in outputs]))
      print("> ", end="")
      sys.stdout.flush()
      sentence = sys.stdin.readline()

def get_src_bucket_id(token_ids, buckets=None, hidden=None):
  if not buckets:
    buckets = model_utils._buckets
  if hidden is None:
    bucket_id = min([b for b in xrange(len(buckets)) if buckets[b][0] >= len(token_ids)])
  else:
    bucket_id = max([b for b in xrange(len(buckets))])
  logging.info("Bucket {}".format(buckets[bucket_id]))
  logging.info("Input: {}".format(token_ids))
  return bucket_id

def get_token_ids(sentence, config):
  token_ids = [int(tok) for tok in sentence.strip().split()]
  token_ids = [w if w < config['src_vocab_size'] else data_utils.UNK_ID for w in token_ids]
  if config['add_src_eos']:
    token_ids.append(data_utils.EOS_ID)
  return token_ids

def get_outputs(session, config, model, sentence, buckets=None, hidden=None):
  token_ids = get_token_ids(sentence, config)
  bucket_id = get_src_bucket_id(token_ids, buckets=buckets, hidden=hidden)
  enc_inputs, dec_inputs, trg_weights, seq_length, src_mask, trg_mask = model.get_batch(
    {bucket_id: [(token_ids, [])]}, bucket_id, config['encoder'])
  _, _, logits, states = model.get_state_step(session, enc_inputs, dec_inputs,
                                              trg_weights, bucket_id, forward_only=True,
                                              sequence_length=seq_length, src_mask=src_mask,
                                              trg_mask=trg_mask, hidden=hidden)
  outputs = []
  for logit in logits:
    outputs.append(int(np.argmax(logit, axis=1)))
    if outputs[-1] == data_utils.EOS_ID:
      break
  return outputs, states

def main(_):
  config = model_utils.process_args(FLAGS, train=False, greedy_decoder=True)
  if FLAGS.interactive:
    decode_interactive(config)
  else:
    decode(config)

def current_time_str():
  return datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S')

if __name__ == "__main__":
  logging.getLogger().setLevel(logging.INFO)
  logging.info("Start: {}".format(current_time_str()))
  tf.app.run()
  logging.info("End: {}".format(current_time_str()))
