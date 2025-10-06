# gender-bias-qe-metrics
Code for the paper: **Watching the Watchers: Exposing Gender Disparities in Machine Translation Quality Estimation**.

Find the paper here
[![ACL Anthology](https://img.shields.io/badge/ACL-Anthology-ED1C24)](https://aclanthology.org/2025.acl-long.1228/)



> [!NOTE]
> This repository is WIP. We will update it shortly.


## 🧠 What this paper studies

- **Context**: While QE metrics have been optimized to align with human judgments, whether they encode social biases has been largely overlooked.
- **Goal**: This paper defines and investigates gender bias of QE metrics and discusses its downstream implications for machine translation (MT).
- **Breakdown**:
  - **Examining bias in gender-ambiguous instances** 

    Denoted as: `ambiguous`
  - **Examining bias in gender-unambiguous instances**
    - **a) Intra-sentential cues**: when within-sentence cues resolve referent ambiguity. 
    
        Denoted as: `non-ambiguous-counterfactual`
    - **b) Extra-sentential cues (non-ambiguous-Contextual)**: when cross-sentence or broader context resolves referent ambiguity. 
    
         Denoted as: `non-ambiguous-contextual`
- **Downstream implications**:
  - **a) Data quality filtering**
  - **b) Machine translation quality assessment**
  - **c) Quality-aware decoding**



## ⚙️ Requirements and Dependencies

Create your virtual environment and install dependencies

```bash
python3 -m venv myvenv
source myvenv/bin/activate
pip install -r requirements.txt
```

## 📂 Our Repository Structure
```
gender-bias-qe-metrics/
├─ COMET/  # Comet implementation
├─ data/  # Data of each dataset
│  └─ ... 
├─ metricx/  # MetricX implementation
├─ notebooks/ # Use to analyse scores and make figs
│  └─ ...            
├─ plots/ # Plots generated
│  └─ ...
├─ results/        # Results 
│  ├─ scores/      # Scores for each setup  
│  │  ├─ ambiguous/
│  │  ├─ nonambiguous-contextual/
│  │  └─ nonambiguous-counterfactual/
|  ├─ stats/    # Meta-analysis from notebooks 
│  └─ translations-qad-ambiguous/ # QAD translations
├─ src/
│  ├─ eval_metrics.py    # Evaluation helpers
│  ├─ utils_metrics.py   # Utils used
│  ├─ qad-reranking/     🧪Pipeline for QAD 
│  ├─ score_ambiguous/   🧪Pipeline ambiguous-case 
│  ├─ score_contextual/  🧪Pipeline contextual-case 
│  └─ score_counterfactual/ 🧪Pipeline counterfactual 
├─ requirements.txt
├─ README.md
└─ LICENSE
```

## 🧪 Replicating Experiments

Before running any of the scripts, make sure you set correctly: 1) your environment path and 2) the root project directory.
   
```bash 
ROOT_DIR=path_to_your_root
Path_to_venv=path_to_your_venv
```


🟣 **Gender-ambiguous instances** `ambiguous`
  
In this setting we experiment with 3 datasets: **MT-GenEval**, **mGente** and **GATE**. 

Please use the following script. 
>💡You can declare the metrics that you want to run on the dataset. Set the $DATASET variable to specify the corresponding dataset to be used.

```bash
./src/score_ambigous/run_scoring.sh
```
Scores will be saved in: `./results/scores/ambiguous/[dataset_name]`

📓 For analysing the scores and drawing conclusions check the corresponding jupyter notebooks in: `notebooks/analysis_ambiguous/`

 🟢 **Gender-unambiguous instances**
  - **a) Intra-sentential cues** `non-ambiguous-counterfactual`
  
    Run the following script to get the scores of the metrics, when assessing unambiguous instances with intra-sentencial cues (Section 4.2.1). 
    
    >💡You can select the metrics used by modifying the file
    ```bash
    ./src/score_counterfactual/run_mtgeneval_ref.sh
    ```
    Scores will be saved in: `./results/scores/nonambiguous-counterfactual/mtgeneval/references`

    📓 For analysing the scores and drawing conclusions check the corresponding jupyter notebook: 
    `notebooks/analysis_counterfactual/analysis-MT-GenEval-counterfactual-references.ipynb`
  - **b) Extra-sentential cues** `non-ambiguous-contextual`
    
    We run metrics, on unambiguous instances with extra-sentencial cues (Section 4.2.2). 
    For this setup we use the MT-GenEval dataset.
    
    We use independent scripts for neural and LLM-based metrics.
    For neural metrics you have two options: translating the additional context or not.

    To run the **neural metrics** use the following script:
    ```bash
    ./src/score_contextual/run_mtgeneval_neural.sh
    ```
    >💡Use `TRANSLATE_CONTEXT=False` if you want to run on untranslated context. 

    >💡For experimenting with translated context use first the `translate-contexts.sh` script to obtain translations, and then run the above with  `TRANSLATE_CONTEXT=True`.

    To run the **LLM-based metrics** use the following script:
    ```bash
    ./src/score_contextual/run_mtgeneval_llm_based.sh
    ```
    Scores will be saved in: `results/scores/nonambiguous-contextual/mtgeneval`

    📓 For analysing the scores and drawing conclusions check the corresponding jupyter notebooks in: `notebooks/analysis_contextual/`

 🔵 **Downstream implications**
  - **a) Data quality filtering**

    This part refers to the discussion in Section 5.1.
  - **b) Machine translation quality assessment**
   
    This part refers to Section 5.2, in which we extend our analysis to investigate whether these metrics demonstrate systematic biases when applied to automatic translations focusing on intra-sentential cases. 
    
    The automated translations are generated by Google Translate and are provided in:
    `/data/mtgeneval/google_translated_counterfactual/`

    To run the metrics on the corresponding translations use the following script: 
     ```bash 
     ./src/score_counterfactual/run_mtgeneval_gt_translations.sh
    ```
    Scores will be saved in: `/results/scores/nonambiguous-counterfactual/mtgeneval/gt-translations`

    📓 For analysing the scores and drawing conclusions check the corresponding jupyter notebook:
    `notebooks/analysis_counterfactual/analysis_MT-GenEval-counterfactual-gt-translations.ipynb`

  - **c) Quality-aware decoding**
  
    This part refers to section 5.3.
    
    Use the following script to get greedy outputs:
    ```bash
    ./src/qad-reranking/generate_translations_greeedy.sh
    ```
    Use the following script to get outputs with quality-aware reranking:
    ```bash
    ./src/qad-reranking/generate_translations_reranking.sh
    ```
    📓 For analysing the scores and drawing conclusions check the corresponding jupyter notebooks in:
    `notebooks/qad_reranking/analysis_reranking.ipynb`

### ✉️ Contact
For questions, feedback, or collaboration inquiries, please contact: emmanouil.zaranis@tecnico.ulisboa.pt or manoszar96@gmail.com

### 📚 Citation

```bibtex
@inproceedings{zaranis-etal-2025-watching,
    title = "Watching the Watchers: Exposing Gender Disparities in Machine Translation Quality Estimation",
    author = "Zaranis, Emmanouil  and
      Attanasio, Giuseppe  and
      Agrawal, Sweta  and
      Martins, Andre",
    editor = "Che, Wanxiang  and
      Nabende, Joyce  and
      Shutova, Ekaterina  and
      Pilehvar, Mohammad Taher",
    booktitle = "Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2025",
    address = "Vienna, Austria",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.acl-long.1228/",
    doi = "10.18653/v1/2025.acl-long.1228",
    pages = "25261--25284",
    ISBN = "979-8-89176-251-0"
}
```

### 🙏 Acknowledgments 

This work was supported by the Portuguese Recovery and Resilience Plan through project C64500888200000055 (Center for Responsible AI), by EU’s Horizon Europe Research and Innovation Actions (UTTER, contract 101070631), by the project DECOLLAGE (ERC-2022-CoG 101088763), and by FCT/MECI through national funds and EU funds under UID/50008: Instituto de Telecomunicações.