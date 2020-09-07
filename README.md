#X-vector pytorch implementation

I'm currently using TIMIT to run the test. Be careful that you have to convert the voices in TIMIT from 'NIST' format to 'RIFF' format first, otherwise the file could not be opened successfully.

mfcc.py
Extract mfcc from voices and store them in .pkl file.

tdnn.py
Define the structure of tdnn.

train.py
Train the network. Usage:type "python -u train.py batch_size learning_rate | tee ~/train_log.txt".
The log of training process will be recorded in "train_log.txt" in the same directory.




