# AICity Track 2 Data Preparation

This repository contains the scripts to reproduce the data preparation pipeline for the AICity 2024 Track 2 competition.

## Prerequisites

The raw data is expected to be in the following directory: `/home/deepzoom/arv-aicity2-data/Park/AICITY2024_Track2_AliOpenTrek_CityLLaVA/data_preprocess/data`

If your data is in a different location, you will need to update the paths in the scripts located in the `src/data_preprocessing` directory.

## Data Preparation

To generate the final dataset, run the following command from the root of this repository:

```bash
make prepare_data
```

This will execute the data preparation pipeline and generate the `final_local_train_dataset.json` file in the `data/` directory.