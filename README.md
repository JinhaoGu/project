#X-vector pytorch implementation

modify_name.py
Create dir "train_spkr" and "test_spkr" which contains training and testing set respectively.
I'm currently using TIMIT to run the test. Be careful that you have to convert the voices in TIMIT from 'NIST' format to 'RIFF' format first, otherwise the file could not be opened successfully.
mfcc.py
Extract mfcc from voices and store them in .pkl file.

tdnn.py
Define the structure of tdnn.

train.py
Train the network (not finished).




