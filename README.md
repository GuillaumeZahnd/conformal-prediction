# Conformal prediction

## Classification tasks

### Example

`uv run examples/demo_classification_fashion_mnist.py`

```sh
----------------------------------------------------------------
CALIBRATION REPORT
----------------------------------------------------------------
Calibration samples: 5000
alpha: 0.05  (target coverage >= 95.0%)
qhat: 0.8748
A class is included in the prediction set if its softmax probability is >= 0.1252
```

```sh
----------------------------------------------------------------
UNCERTAINTY REPORT
----------------------------------------------------------------
Coverage (true label in prediction set): 4786/5000 (95.7%)

Set size summary: mean 1.46, median 1, 90% of samples in [1, 3], range [1, 5]

set size |   count | frequency | coverage
       1 |    3184 |     0.637 |    0.967
       2 |    1354 |     0.271 |    0.938
       3 |     428 |     0.086 |    0.946
       4 |      33 |     0.007 |    0.970
       5 |       1 |     0.000 |    1.000
```

### Calibration procedure

<img width="1999" height="1341" alt="classification_calibration" src="https://github.com/user-attachments/assets/51c200c2-bd88-4481-9a2c-c3913e961aff" />

### Uncertainty quantification

For a new input $x$, the prediction set $\mathcal{S}(x) \subseteq \mathcal{Y}$ collects every candidate class $y$ from the label space $\mathcal{Y}$ whose conformity score satisfies the pre-calibrated threshold $\hat{q}$. This guarantees marginal coverage $P\Big(Y \in \mathcal{S}(X)\Big) \ge 1 - \alpha$ without knowing the true label $Y$. The set size $\vert\mathcal{S}(x)\vert$ indicates the model uncertainty, for a specific sample $x$ and coverage level $\alpha$.

- Singleton set ($\vert\mathcal{S}(x)\vert=1$): High confidence (a single clear candidate class).
- Larger set ($\vert\mathcal{S}(x)\vert \geq 2$): Lower confidence (ambiguity among multiple candidates classes).
- Empty set ($\vert\mathcal{S}(x)\vert=0$): High uncertainty or out-of-distribution (no class meets the confidence threshold).

Across an entire dataset, the distribution of set sizes $\vert\mathcal{S}(X)\vert$ indicates the model's overall uncertainty profile at coverage level $\alpha$.

<img width="1999" height="1334" alt="classification_results" src="https://github.com/user-attachments/assets/0cc14a89-fb9a-40ef-aac8-9d60aedd78d4" />

## Regression tasks

### Example

`uv run examples/demo_regression_california_housing.py`

```sh
----------------------------------------------------------------
CALIBRATION REPORT
----------------------------------------------------------------
Calibration samples: 3096
alpha: 0.05  (target coverage >= 95.0%)
qhat: 1.1620  (in the units of the target)
prediction interval: prediction +/- 1.1620
```

```sh
----------------------------------------------------------------
UNCERTAINTY REPORT
----------------------------------------------------------------
Target coverage:    >= 95.0% (alpha = 0.05)
Empirical coverage: 2961/3096 (95.6%)
Interval width:     2.3241  (identical for every sample)
```

## Resources

- AN Angelopoulos and S Bates. [**"Conformal prediction: A gentle introduction."**](https://arxiv.org/abs/2107.07511) Foundations and Trends in Machine Learning, 16(4):494-591, 2023.

