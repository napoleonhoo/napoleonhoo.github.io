---
layout: default
---

# 模型中各个指标的含义与计算

Abaucs/Paddle/Flux中常见的一条日志

> item_sim: auc=0.973279 bucket_error=1.10328 rmse=0.418493 num=17283180 mae=0.340888 actural_ctr=0.180089 predict_ctr=0.482359 copc=0.373351

## 1 AUC

### 1.1 含义与公式

参见另一篇文章：[ROC曲线与ROC AUC](./roc_auc.html)

### 1.2 计算

参见abacus代码：

``` cpp
    _table[label][std::min(int(pred * _table_size), _table_size - 1)]++;

    double area = 0;
    double fp = 0;
    double tp = 0;

    for (int i = _table_size - 1; i >= 0; i--) {
        double newfp = fp + table[0][i];
        double newtp = tp + table[1][i];
        area += (newfp - fp) * (tp + newtp) / 2;
        fp = newfp;
        tp = newtp;
    }

    _auc = area / (fp * tp);
```

## 2 Bucket Error

### 代码

``` cpp
void AucCalculator::calculate_bucket_error() {
    double last_ctr = -1;
    double impression_sum = 0;
    double ctr_sum = 0.0;
    double click_sum = 0.0;
    double error_sum = 0.0;
    double error_count = 0;
    double* table[2] = {&_table[0][0], &_table[1][0]};
    for (int i = 0; i < _table_size; i++) {
        double click = table[1][i];
        double show = table[0][i] + table[1][i];
        double ctr = (double)i / _table_size;
        if (fabs(ctr - last_ctr) > kMaxSpan) {
            last_ctr = ctr;
            impression_sum = 0.0;
            ctr_sum = 0.0;
            click_sum = 0.0;
        }
        impression_sum += show;
        ctr_sum += ctr * show;
        click_sum += click;
        double adjust_ctr = ctr_sum / impression_sum;
        double relative_error = sqrt((1 - adjust_ctr) / (adjust_ctr * impression_sum));
        if (relative_error < kRelativeErrorBound) {
            double actual_ctr = click_sum / impression_sum;
            double relative_ctr_error = fabs(actual_ctr / adjust_ctr - 1);
            error_sum += relative_ctr_error * impression_sum;
            error_count += impression_sum;
            last_ctr = -1;
        }
    }
    _bucket_error = error_count > 0 ? error_sum / error_count : 0.0;
```

## 3 Absolute Error & Squre Error

### 公式

$$
abserr = \sum{|pred - label|}
$$

$$
sqrerr = \sum{(pred - label)^2}
$$

### 代码

``` cpp
    _local_abserr += fabs(pred - label);
    _local_sqrerr += (pred - label) * (pred - label);
```

## 4 RMSE

### 公式

$$
rmse = \frac{sqrerr}{fp + tp}
$$

### 代码

``` cpp
    _rmse = sqrt(abacus::mpi_allreduce(_local_sqrerr, MPI_SUM) / (fp + tp));
```

## 5 MAE

### 公式

$$
mae = \frac{abserr}{fp + tp}
$$

### 代码

``` cpp
    _mae = abacus::mpi_allreduce(_local_abserr, MPI_SUM) / (fp + tp);
```

## 6 Actual CTR

### 公式

$$
ActualCTR = \frac{tp}{tp + fp}
$$

### 代码

``` cpp
    _actual_ctr = tp / (fp + tp);
```

## 7 Predict CTR

### 公式

$$
PredictCTR = \frac{\sum{pred}}{tp + fp}
$$

### 代码

``` cpp
    _predicted_ctr = abacus::mpi_allreduce(_local_pred, MPI_SUM) / (fp + tp);
```

## 8 COPC (Click Over Predicted Click)

### 公式

$$
COPC = \frac{ActualCTR}{PredictCTR}
$$

### 代码

``` cpp
    copc = _actual_ctr / _predicted_ctr;
```

## PN Targets

### 日志举例

> res_one_click_join_pn: Average_PN=5.44591 ins_number=7982670 positive_pairs=45945386 negative_paris=8436670

### 算法内容

对不同的uid内，以label为基准，对{label, pred}进行排序。当label0 < label1时，如果pred0 < pred1，则为**正序**（positive_pairs，PosNum）；否则，为**逆序**（negative pairs，NegNum）。有：

$$
pn = \frac{PosNum}{PosNum + NegNum}
$$

## WUAUC

### 日志举例

> t7_order_wuauc_join: Tag=all WUAUC=0.835354 UAUC=0.829737 LogLoss=0.506748 ActualCTR=0.498121 PredictedCTR=0.50286 COPC=0.990576 RealUserCount=1995877 RealInsNum=5454408 ValidUserCount=411705 ValidInsNum=2844012

### 算法内容

在不同Tag下，对不同的UID，分别计算AUC。对不同的tag分别有：

$$
RealUserCount = length\ of\ set(\mathbb{UID}) \qquad \text{, if } InsNum_i \ge 0
\\
RealInsNum = \sum_i{InsNum_i} \qquad \text{, if } InsNum_i \ge 0
$$


$$
ValidUserCount = length\ of\ set(\mathbb{UID}) \qquad \text{, if } AUC_i \ne -1
\\
RealUserCount = \sum_i{InsNum_i} \qquad \text{, if } AUC_i \ne -1
$$


$$
LogLoss =
    \begin{cases}
        \sum_i{-1 \times \log(pred_i)}       & \quad \text{, if label = 1}
        \\
        \sum_i{-1 \times \log(1 - pred_i)}   & \quad \text{, if label = 0}
    \end{cases}
\\
LogLossSum = \sum_i{LogLoss_i} \qquad \text{, if } InsNum_i \ge 0
\\
LogLoss = LogLossSum \div RealInsNum
$$


$$
PredictedCTRSum = \sum_i{PredictedCTR_i} \qquad \text{, if } InsNum_i \ge 0
\\
PredictedCTR = PredictedCTRSum \div RealInsNum
$$


$$
ActualCTRSum = \sum_i{ActualCTR_i} \qquad \text{, if } InsNum_i \ge 0
\\
ActualCTR = ActualCTRSum \div RealInsNum
$$


$$
COPC = ActualCTR \div PredictedCTR
$$


$$
UAUCSum = \sum_i{AUC_i} \qquad \text{, if } AUC_i \ne -1
\\
UAUC = UAUCSum \div ValidUserCount
$$


$$
WUAUCSum = \sum_i{AUC_i \times InsNum_i} \qquad \text{, if } AUC_i \ne -1
\\
WUAUC = WUAUCSum \div ValidInsNum
$$

WUAUC类似GAUC（阿里研发的）。而GAUC的公式有：

$$
\begin{align*}
GAUC= & \frac{\sum_{(i, p)} (w_{(i,p)} \times AUC_{i, p})}{\sum_{(i, p)} w_{(i,p)}} \\
      & \frac{\sum_i (w_i \times AUC_i)}{\sum_i w_i} \\
      & \frac{\sum_i (Impression_i \times AUC_i)}{\sum_i Impression_i}
\end{align*}
$$

## Pairwise Ratio

日志举例

> click_ins_ratio: Ins=78998676 Same_label_click=13648034 Left_label_click=17479466 Right_label_click=17364220 Same_label_unclick=30506956

具体逻辑如下，对于两个不同的lable来说有：

$$
\begin{align*}
    & SameLabelClick \leftarrow SameLabelClick + 1, & \text{ if } label_i = 1\ and\ label_j = 1
\\
    & LeftLabelClick \leftarrow LeftLabelClick + 1, & \text{ if } label_i = 1\ and\ label_j = 0
\\
    & RightLabelClick \leftarrow RightLabelClick + 1, & \text{ if } label_i = 0\ and\ label_j = 1
\\
    & SameLabelUnclick \leftarrow SameLabelUnclick + 1, & \text{ if } label_i = 1\ and\ label_j = 0
\end{align*}
$$
