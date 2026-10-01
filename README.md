# Conformal prediction

## How to interpret the results

### Classification tasks

For a new input $x$, the prediction set $\mathcal{S}(x) \subseteq \mathcal{Y}$ collects every candidate class $y$ from the label space $\mathcal{Y}$ whose conformity score satisfies the pre-calibrated threshold $\hat{q}$. This guarantees marginal coverage $P\Big(Y \in \mathcal{S}(X)\Big) \ge 1 - \alpha$ without knowing the true label $Y$. The set size $\vert{}\mathcal{S}(x)\vert{}$ is indicative of the model uncertainty, for a specific sample $x$ and coverage level $\alpha$.

- Singleton set ($\vert{}\mathcal{S}(x)\vert{}=1$): High confidence (a single clear candidate class).
- Larger set ($\vert{}\mathcal{S}(x)\vert{} \geq 2$): Lower confidence (ambiguity among multiple candidates classes).
- Empty set ($\vert{}\mathcal{S}(x)\vert{}=0$): High uncertainty or out-of-distribution (no class meets the confidence threshold).

Across an entire dataset, the distribution of set sizes $\vert{}\mathcal{S}\vert{}$ indicates the model's overall uncertainty profile at coverage level $\alpha$.

## Resources

- AN Angelopoulos and S Bates. [**"Conformal prediction: A gentle introduction."**](https://arxiv.org/abs/2107.07511) Foundations and Trends in Machine Learning, 16(4):494-591, 2023.
