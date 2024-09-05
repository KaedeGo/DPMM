import pandas as pd
seed = 42

if __name__ == "__main__":
    stay_set = set()
    for task in ['in-hospital-mortality', 'phenotyping_48h', 'readmission']:
        for split in ['train', 'val', 'test']:
        # for split in ['test']:
            file_path = f'/home/fwu/Documents/myProjects/MedFuse/mimic3extract/data/{task}/{split}_listfile.csv'
            data = pd.read_csv(file_path)
            sorted_data = data.sort_values(by='stay', ascending=True)
            print(f'Number of stays in {task} {split}:', len(sorted_data))
            sampled_data = sorted_data.sample(frac=1/3, random_state=seed) 
            print(f'Number of stays in {task} {split} after sampling:', len(sampled_data))
            stay_set.update(sampled_data['stay'])
            print('Number of unique stays:', len(stay_set))
            sampled_data.to_csv(f'/home/fwu/Documents/myProjects/MedFuse/mimic3extract/data/{task}/{split}_note_listfile.csv', index=False)