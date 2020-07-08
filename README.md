# CoLA
A Trandformer style and a BiLSTM with ELMo to evaluate on Corpus of Linguistic Acceptibility(CoLA dataset)
## Dataset
The dataset can be downloaded [here](https://nyu-mll.github.io/CoLA/)
## Training
To train and validate the model run 
```
python train.py
```
Optional Arguments
```
--batch_size              Batch size to train the model           
--lr                      Learning rate for the model
--hidden_size             Hidden size, either for the BiLSTM for the Transformer
--embed_size              Embedding size, either for the BiLSTM for the Transformer
--n_heads                 Number of heads for Multi Head Attention for the Traansformer
--n_layer                 Number of layers in the Transformer
--per_layer               Iterations per layer of the Transformer
--inter_size              Intermediate size for the Transformer
--train_path              Path to the training file
--dev_path                Path to the development file
--epochs                  Number of epochs for training
--min_word_count          Minimum word count to include a word in the vocabulary
--eval_every              Evaluate after how many epochs
--dropout_prob_classifier Dropout rate for the Classifier in the Transformer
--dropout_prob_attn       Dropout rate for self attention 
--dropout_prob_hidden     Dropout rate for the self feed forward network of the Transformer
--num_rep                 Number of ELMo representations
--elmo_drop               Dropout rate for ELMo
--use_elmo                Whether to use ELMo or not
--BiLSTM                  Whether to use BiLSTM model or not
--transformer             Whether to use Transformer model or not
```
        
