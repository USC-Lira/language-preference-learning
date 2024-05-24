# Data Generation

## WidowX Experiment
1. `python read_real_data.py`: preprocess the raw data and put them into `data_avoid_danger/`
2. `python split_traj.py`: split the trajectories into train, val and test set
3. `python make_dataset_avoid_danger.py --data-dir=...`: generate train, val and test set based on the split results
4. `bash scripts/lang_preprocess.sh`: preprocess the language comparisons
5. `bash scripts/resize.sh`: resize image observations to 224x224