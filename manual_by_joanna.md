Manual can also be found at: http://etherpad.mit.edu/p/the_very_important_manual

Code available at: https://github.com/jstadnik/cam-tf-seq2seq at branch feed_aligns

For code repo located in folder /home/miproj/4thyr.oct2017/js2166/code/cam-tf-seq2seq:
    
=====================================================================
    
To train:
        
#!/bin/bash
 #$ -S /bin/bash
 
 echo "jobid="$JOB_ID
 
 
 config=/data/mifs_scratch/js2166/bpe/wat.bpe2bpe.seqlen80.ini
 data_dir=/data/mifs_scratch/js2166/bpe/data
 train_dir=/data/mifs_scratch/js2166/bpe/train_align_0.1_fixed
 script_dir=/home/miproj/4thyr.oct2017/js2166/code/cam-tf-seq2seq
 
 export seq2seq=$script_dir
 export PYTHONPATH=$seq2seq:$PYTHONPATH
 
 source /home/miproj/4thyr.oct2017/js2166/tf_original_helper.sh
 
INSERT COMMAND FUNCTION HERE
 
 mkdir -p $train_dir
 cp $config $train_dir
 cmd="$cmd --steps_per_checkpoint=2000 --device=$device"
 echo "cmd="$cmd; $cmd
 
 
COMMAND FUNCTION VARIANTS:
1) Normal train no alignment stuff
 cmd="python $script_dir/cam_tf_alignment/train.py --train_dir=$train_dir --config_file=$config -- train_src_idx=$data_dir/train.bpe.ja --train_trg_idx=$data_dir/train.bpe.en --dev_src_idx=$data_dir/dev. bpe.ja --dev_trg_idx=$data_dir/dev.bpe.en --max_to_keep=2"
2) Train with reducing attention entropy, scaling factor lamb=0.1
 cmd="python $script_dir/cam_tf_alignment/train.py --train_dir=$train_dir --config_file=$config -- train_src_idx=$data_dir/train.bpe.ja --train_trg_idx=$data_dir/train.bpe.en --dev_src_idx=$data_dir/dev. bpe.ja --dev_trg_idx=$data_dir/dev.bpe.en --max_to_keep=2 --lamb=0.1"
3) Train with MSE or xent alignment loss, scaling factor lamb=0.1
 cmd="python $script_dir/cam_tf_alignment/train.py --train_dir=$train_dir --config_file=$config -- train_src_idx=$data_dir/train.bpe.ja --train_trg_idx=$data_dir/train.bpe.en --dev_src_idx=$data_dir/dev. bpe.ja --dev_trg_idx=$data_dir/dev.bpe.en --max_to_keep=2 --train_align=$data_dir/bpe_aligns_train_norm --entropy=0.1"
 
To determine MSE or xent go to 
/home/miproj/4thyr.oct2017/js2166/code/cam-tf-seq2seq/cam_tf_alignment/seq2seq/seq2seq.py and uncomment the correct line in sequence_loss_by_example
bpe_aligns_train_norm is a pickle containing normalised bpe matrices for every word (appropriately padded in deq2seq_model.py in create_batch). I created them using wrd2bpe2.py and then a normalise script I believe, but might have messed it up. See in utils/data_utils to see what makes sense
 
=========================================================================

To force-decode and get attentions:
    
    
 #!/bin/bash
 #$ -S /bin/bash
 
 echo "jobid="$JOB_ID
 
 
 data_dir=/data/mifs_scratch/js2166/bpe/data
 model_dir=/data/mifs_scratch/js2166/bpe/train_align_xent_0.1_fixed
 config=/data/mifs_scratch/js2166/bpe/wat.bpe2bpe.seqlen80.ini
 output_dir=$model_dir/test_results
 script_dir=/home/miproj/4thyr.oct2017/js2166/code/cam-tf-seq2seq
 
 export seq2seq=$script_dir
 export PYTHONPATH=$seq2seq:$PYTHONPATH
 
 source /home/miproj/4thyr.oct2017/js2166/tf_original_helper.sh
 
 cmd="python $script_dir/cam_tf_alignment/decode_feed.py --model_path=$model_dir/train.ckpt-       dev_bleu --config_file=$config --test_src_idx=$data_dir/test.bpe.ja --test_out_idx=$output_dir/ignore.   test.out.bpe --atts_out=$output_dir/test.atts.out.bpe --max_sentences=-1 --trg_idx=$data_dir/test.bpe.en "
 
 cmd="$cmd --device=$device"
 echo "cmd="$cmd; $cmd
 
 
======================================================================

To calculate both AERs and get pretty graphs:
Move all the .e00000000 and .o00000000 output files to the correct training dir and concat them into 'eugh' and 'ough'
Then in the training dir: /home/mifs/ds636/code/shell-tools/log_bleu.pl eugh  > eugh_short
Then: python /helper_files/align/combined.py path_to_training_dir -1
(last digit corresponds to how many checkpoints you want to use, -1 ==> all of them)

=====================================================================

To just get AER:
python /helper_files/align/aer5.py dev path_to_training_dir/atts_output_file
replace 'dev' with 'test' to run on test set
replace are5.py with aer6.py to calculate using all above threshold (aer5.py uses argmax)

======================================================================

To plot losses:
python helper_files/align/plot_losses2.py path_to_training_dir/ough


