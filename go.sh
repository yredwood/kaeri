

seed=$1
gpu=$2
model_name=jiyu2_${seed}

python preprocessor.py $seed
#python xgb.py $model_name $seed
CUDA_VISIBLE_DEVICES=$gpu python train.py $model_name $seed


#for (( seed=0;seed<12;seed+=1 ))
#do
#    xgb_name=xgb_${seed}
#    python xgb.py $xgb_name $seed
#done
