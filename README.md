# Conformal prediction

## Calibration procedure

### Classification tasks

<img width="1999" height="1341" alt="classification_calibration" src="https://github.com/user-attachments/assets/51c200c2-bd88-4481-9a2c-c3913e961aff" />

## How to interpret the results

### Classification tasks

For a new input $x$, the prediction set $\mathcal{S}(x) \subseteq \mathcal{Y}$ collects every candidate class $y$ from the label space $\mathcal{Y}$ whose conformity score satisfies the pre-calibrated threshold $\hat{q}$. This guarantees marginal coverage $P\Big(Y \in \mathcal{S}(X)\Big) \ge 1 - \alpha$ without knowing the true label $Y$. The set size $\vert\mathcal{S}(x)\vert$ indicates the model uncertainty, for a specific sample $x$ and coverage level $\alpha$.

- Singleton set ($\vert\mathcal{S}(x)\vert=1$): High confidence (a single clear candidate class).
- Larger set ($\vert\mathcal{S}(x)\vert \geq 2$): Lower confidence (ambiguity among multiple candidates classes).
- Empty set ($\vert\mathcal{S}(x)\vert=0$): High uncertainty or out-of-distribution (no class meets the confidence threshold).

Across an entire dataset, the distribution of set sizes $\vert\mathcal{S}(X)\vert$ indicates the model's overall uncertainty profile at coverage level $\alpha$.

<img width="1999" height="1334" alt="classification_results" src="https://github.com/user-attachments/assets/0cc14a89-fb9a-40ef-aac8-9d60aedd78d4" />

## Resources

- AN Angelopoulos and S Bates. [**"Conformal prediction: A gentle introduction."**](https://arxiv.org/abs/2107.07511) Foundations and Trends in Machine Learning, 16(4):494-591, 2023.

