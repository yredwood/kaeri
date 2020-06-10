

seed=$1
model_name=rnn_${seed}

#python preprocessor.py $seed
#python xgb.py $model_name $seed
CUDA_VISIBLE_DEVICES=2 python train.py $model_name $seed


#for (( seed=0;seed<12;seed+=1 ))
#do
#    xgb_name=xgb_${seed}
#    python xgb.py $xgb_name $seed
#done
