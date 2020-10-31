# NLP-Toxic-Comment-Classification
Kaggle Competition

Naive Bayse Model as Baseline, result as follow:

2020-10-24 21:59:00,770 accuracy:0.9066583111389629


              precision    recall  f1-score   support

           0       0.85      0.58      0.69      3101
           1       0.42      0.37      0.39       329
           2       0.79      0.57      0.66      1698
           3       0.16      0.03      0.05        91
           4       0.71      0.51      0.60      1594
           5       0.35      0.12      0.18       298

   micro avg       0.76      0.53      0.62      7111
   macro avg       0.55      0.36      0.43      7111
weighted avg       0.75      0.53      0.62      7111
 samples avg       0.98      0.94      0.93      7111

CNN: python run_cnn.py --config config/config.yaml

Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
embedding (Embedding)        (None, 128, 100)          2000000   
_________________________________________________________________
conv1d (Conv1D)              (None, 128, 128)          89728     
_________________________________________________________________
batch_normalization (BatchNo (None, 128, 128)          512       
_________________________________________________________________
max_pooling1d (MaxPooling1D) (None, 64, 128)           0         
_________________________________________________________________
batch_normalization_1 (Batch (None, 64, 128)           512       
_________________________________________________________________
conv1d_1 (Conv1D)            (None, 64, 256)           164096    
_________________________________________________________________
global_max_pooling1d (Global (None, 256)               0         
_________________________________________________________________
batch_normalization_2 (Batch (None, 256)               1024      
_________________________________________________________________
dense (Dense)                (None, 256)               65792     
_________________________________________________________________
dropout (Dropout)            (None, 256)               0         
_________________________________________________________________
batch_normalization_3 (Batch (None, 256)               1024      
_________________________________________________________________
dense_1 (Dense)              (None, 6)                 1542      
_________________________________________________________________
dense_2 (Dense)              (None, 6)                 42        
_________________________________________________________________
Total params: 2,324,272
Trainable params: 2,322,736
Non-trainable params: 1,536
_________________________________________________________________
3990/3990 [==============================] - 245s 61ms/step - loss: 0.1016 - accuracy: 0.9401
2020-10-28 21:07:18,059 accuracy:0.9048409838633871
2020-10-28 21:07:18,060 
              precision    recall  f1-score   support

           0       0.82      0.58      0.68      3101
           1       0.34      0.55      0.42       329
           2       0.77      0.71      0.74      1698
           3       1.00      0.00      0.00        91
           4       0.68      0.64      0.66      1594
           5       0.35      0.24      0.29       298

   micro avg       0.71      0.60      0.65      7111
   macro avg       0.66      0.46      0.47      7111
weighted avg       0.74      0.60      0.65      7111
 samples avg       0.98      0.95      0.93      7111

cnn经验：

• multi-classes single-label task:
  • softmax (sum = 1)
  • categorical_crossentropy
  
• multi-classes multi-label task，每个分类是独⽴的：
  • ‘sigmoid’ instead of ‘softmax’
  • ‘binary_crossentropy’ instead of ‘categorical_crossentropy
  

Pretrained Embedding:
======================

Learned from training process:
===========================================

twitter is more suitable than wiki

27B meaning records 27billion user data

100d meaning vector size is 100

27B 100d twitter is more suitable than 27B 200d

Using full train set better than partial train set

Best kaggle is 0.94 so far
