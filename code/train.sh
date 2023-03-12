python train_1st.py --config_path ./../config/NFL_effnet-b0-TSM-end_LSTM_32-32_cp-3wh_few_128_drop04_flow_aug.yaml
python train_1st.py --config_path ./../config/NFL_effnet-b0-TSM-end_LSTM_64-64_cp-3wh_few_128_drop06_flow_aug.yaml
python train_1st.py --config_path ./../config/NFL_resnext50-TSM_LSTM_16-16_cp-3wh_few_128_drop04_flow_aug.yaml 
python train_1st.py --config_path ./../config/NFL_resnext50-TSM_LSTM_32-32_cp-3wh_few_128_drop04_flow_aug.yaml
python train_1st.py --config_path ./../config/NFL_resnext50-TSM_LSTM_64-64_cp-3wh_few_128_drop04_flow_aug.yaml

python train_2nd.py --config_path ./../config/NFL_effnet-b0-TSM-end_LSTM_32-32_cp-3wh_few_128_drop04_flow_aug_2nd_lgbm.yaml
python train_2nd.py --config_path ./../config/NFL_effnet-b0-TSM-end_LSTM_64-64_cp-3wh_few_128_drop06_flow_aug_2nd_lgbm.yaml
python train_2nd.py --config_path ./../config/NFL_resnext50-TSM_LSTM_16-16_cp-3wh_few_128_drop04_flow_aug_2nd_xgboost.yaml
python train_2nd.py --config_path ./../config/NFL_resnext50-TSM_LSTM_32-32_cp-3wh_few_128_drop04_flow_aug_2nd_lgbm.yaml
python train_2nd.py --config_path ./../config/NFL_resnext50-TSM_LSTM_64-64_cp-3wh_few_128_drop04_flow_aug_2nd_lgbm.yaml
python train_2nd.py --config_path ./../config/NFL_resnext50-TSM_LSTM_64-64_cp-3wh_few_128_drop04_flow_aug_2nd_xgboost.yaml

python postprocess.py --config_path ./../config/NFL_postprocess.yaml
