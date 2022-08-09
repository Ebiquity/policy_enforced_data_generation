# PriveTab
This repository contains the implementation of <a href="https://ebiquity.umbc.edu/paper/html/id/1014/PriveTAB-Secure-and-Privacy-Preserving-sharing-of-Tabular-Data">PriveTAB : Secure and Privacy-Preserving sharing of Tabular Data</a>. 
We use <a href="https://arxiv.org/abs/1907.00503">CTGAN</a> as our primary generator model. CTGAN can be replaced with any conditional generator model. 
The trained generator model is sampled to create a privacy-preserving synthetic dataset. The sampling process is bound 
by the principle of <a href="https://personal.utdallas.edu/~muratk/courses/privacy08f_files/tcloseness.pdf" target="_blank">t-closeness</a>, ensuring privacy in the shared data

## RUNNING THE CODE
1. To run the code, add the input data to the package. The input data should be of csv format.
2. Specify the discrete attributes of the dataset as "discrete_attributes" in "attributes.ini"
3. Run the code as:
<pre>
<code>
python main.py {path_to_input} {t_value}
</code>
</pre>

5. The t_value argument is optional. The default t-value is 0.6. 
Change the t_value to ensure that the synthetic data is {t_value}-close w.r.t. the original

## RESULTS
The trained generator model is saved in the directory "trained_model". The Synthetic Table is saved in "synthetic_table/synthetic_data.csv".

## CONTACT
For questions or comment, you can reach out to:
Anantaa Kotal (anantak1@umbc.edu)


## REFERENCE

If you use our model(s) or want to refer to it in a paper, please cite:

<pre>
Kotal, Anantaa, et al. "PriveTAB: Secure and Privacy-Preserving sharing of Tabular Data." Proceedings of the 2022 ACM on International Workshop on Security and Privacy Analytics. 2022.
</pre>

## BibTeX ##

<pre>
<code>
@inproceedings{kotal2022privetab,
  title={PriveTAB: Secure and Privacy-Preserving sharing of Tabular Data},
  author={Kotal, Anantaa and Piplai, Aritran and Chukkapalli, Sai Sree Laya and Joshi, Anupam},
  booktitle={Proceedings of the 2022 ACM on International Workshop on Security and Privacy Analytics},
  pages={35--45},
  year={2022}
}
</code>
</pre>
