
# Requirement
* Python3.6
* PyTorch: 1.0.1

# Train models
* Download data and word embedding
* Run the script:
```
nohup python -u joint_main.py --use_pre_trained_model False --use_crf true 
--hidden_dim 300 --attention_query_input_size 303 --num_attention_head 5 
--iteration 1000 --learning_rate 0.02 >log.txt 2>&1 &
```



# Acknowledgments
[NCRF++](https://github.com/jiesutd/NCRFpp)
