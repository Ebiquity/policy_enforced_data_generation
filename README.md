# PriveTab
This repository contains the implementation of "PriveTAB : Secure and Privacy-Preserving sharing of Tabular Data". 
We use CTGAN as our primary generator model. CTGAN can be replaced with any conditional generator model. 
The trained generator model is sampled to create a privacy-preserving synthetic dataset. The sampling process is bound 
by the principle of t-closeness, ensuring privacy in the shared data

## RUNNING THE CODE
1. To run the code, add the input data to the package. The input data should be of csv format.
2. Specify the discrete attributes of the dataset as "discrete_attributes" in "attributes.ini"
3. Run the code as `python main.py <path_to_input> <t_value>`. 
4. The t_value argument is optional. The default t-value is 0.6. 
Change the t_value to ensure that the synthetic data is <t_value>-close w.r.t. the original

## RESULTS
The trained generator model is saved in the directory "trained_model". The Synthetic Table is saved in "synthetic_table/synthetic_data.csv".

## CONTACT
For questions or comment, you can reach out to:
Anantaa Kotal (anantak1@umbc.edu)

## REFERENCE
If you use this code, please cite the following paper:
@inproceedings{kotal2022privetab,
  title={PriveTAB: Secure and Privacy-Preserving sharing of Tabular Data},
  author={Kotal, Anantaa and Piplai, Aritran and Chukkapalli, Sai Sree Laya and Joshi, Anupam},
  booktitle={Proceedings of the 2022 ACM on International Workshop on Security and Privacy Analytics},
  pages={35--45},
  year={2022}
}
