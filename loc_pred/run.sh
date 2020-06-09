
seed=11
model_name=resub_${seed}

#cd data
#python _static_processor.py $seed
#
#cd ..
#CUDA_VISIBLE_DEVICES=1 python train.py $model_name $seed


for (( seed=0;seed<12;seed+=1 ))
do
    xgb_name=xgb_${seed}
    python xgb.py $xgb_name $seed
done
