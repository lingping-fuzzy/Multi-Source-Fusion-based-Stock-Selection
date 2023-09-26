# Multi-Source-Fusion-based-Stock-Selection

This code is the implementation of the Multi-Source-Fusion-based-Stock-Selection paper: 

If you have a problem understanding some code, please also refer to [code](https://github.com/jimmyg1997/NTUA-Multi-Criteria-Decision-Analysis) from co-author ```jimmyg1997```

> **A generalization of multi-source fusion-based framework to stock selection**\
> Václav Snášel, Juan D. Velásquez, Millie Pant, Dimitrios Georgiou, Lingping Kong\
> VSB-Technical university of Ostrava\
> Paper: [https://arxiv.org/abs/2106.03306](https://www.sciencedirect.com/science/article/pii/S1566253523003342)
>
> ![Data fusion](/images/datafusion.png " multi-source fusion")

> **Abstract.** Selecting outstanding technology stocks for investment is challenging. Specifically, the research of the investment for academic purposes is not mature enough due to the disarray of the publications and the overwhelming informative experience from profit-making websites. Often, some authors entitle the stock price prediction as a stock selection problem; both prediction and selection are just sub-sections of portfolio management. Moreover, stock websites provide numerous potential criteria showing various evaluations to simulate and monitor the stock market professionally, which increases the difficulty of academic studies on stock selection. 

The paper generalizes a novel framework with a user-interactive interface for stock selection problems based on multi-source data fusion and decision-level fusion to enhance reliability limited by the narrow criteria performance and the strength of a model overcoming the weakness of a single-performed model. This framework benefits the time-series prediction and decision-making study. Besides, we propose adopting dynamic time warping to assist a task-learning process by customizing a loss function that improves the accuracy of data prediction. The experiment shows that the proposed method reduces the prediction log error by 6.3\% on average and decreases the warping cost by 5.6\% on average over all cases of real-situation data. Finally, we illustrate the proposed framework by implementing it in a real-world stock data selection. The results are practical and effective, further justified through a detailed ablation study. 

### Main script

### Examples

#### 1. Run finance_mcdm.ipynb with the initial data from the data folder:
  Remember to replace your API key to use Alpha vintage data. For details, refer to the readme file.
#### 2. Run xgboost_pred.py, when the predicted time-series stock price data is obtained (during a step in the middle of  finance_mcdm.ipynb)
 then other .py files are supporters or analysis utils.
 #### 3. Run the rest of the process MCDM in finance_mcdm.ipynb
 
### Datasets
 you could download data from investing.com directly for the ```fundamental```, ```analysis``` and ```performance``` Excel data, or you could use the data we provided in the data folder. 
 You could directly use code from finance_mcdm.ipynb for the yahoo finance data and Alpha vintage data.

 ## Citation
If you use this codebase, or otherwise found our work valuable, please cite:
```
@article{snavsel2023generalization,
  title={A generalization of multi-source fusion-based framework to stock selection},
  author={Sn{\'a}{\v{s}}el, V{\'a}clav and Vel{\'a}squez, Juan D and Pant, Millie and Georgiou, Dimitrios and Kong, Lingping},
  journal={Information Fusion},
  pages={102018},
  year={2023},
  publisher={Elsevier}
}
```
 
