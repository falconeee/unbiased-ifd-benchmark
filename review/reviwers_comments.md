Reviewer #2: This manuscript investigates whether architectural depth reliably predicts generalization in vibration-based intelligent fault diagnosis when similarity bias is mitigated through operating-condition-separated evaluation. The topic is relevant, and the effort to promote more rigorous evaluation protocols is valuable. The manuscript is generally readable, and the use of multiple public datasets is potentially useful. However, the present study has fundamental limitations in novelty, experimental design, and evidential support. In particular, several central conclusions are substantially stronger than what the reported experiments can establish.
2. The evidence does not sufficiently support the conclusion that "architectural depth does not guarantee generalization." This conclusion mainly derives from a comparison between AlexNet and ResNet-18 on the so-called "most challenging dataset." However, the two architectures differ not only in depth but also in convolutional kernels, receptive fields, pooling operations, parameter counts, and regularization strategies. Therefore, the manuscript cannot attribute their performance difference solely to network depth. Drawing such a conclusion requires a controlled comparison that isolates architectural depth while holding other variables constant.
3. Similarly, the conclusion that "a larger receptive field leads to greater robustness" remains speculative. The manuscript does not present any controlled experiments on convolutional kernel size or effective receptive field to support this claim.


Coautor Comment: O principal problema a ser resolvido no artigo é o da hipótese 2. Eu concordo completamente com a critica do revisor 2. Nossa conclusao nao se sustenta com base nos resultados. Só em PU que essa possibilidade se confirmou. Isso nao aconteceu nos datasets do CWRU e do MFPT. Além disso, há as outras diferencas de arquitetura que podem ter sido responsaveis por isso. Definitivamente, temos que fazer os ablation studies sugeridos pelo Claude. Tanto o de depth quanto o de receptive fields. Podemos usar a CNN-1D para isso pois deve ser mais simples de modificar e mais rapido de executar, certo? Eu faria os estudos no PU e também nos dois CWRU. Aí temos que torcer que consigamos argumentos para confirmar a hipotese (se nao der, podemos tentar outros datasets, mas nao vejo isso como necessidade).

**Comment R2.2:** *"The evidence does not sufficiently support the conclusion that 'architectural depth does not guarantee generalization.' This conclusion mainly derives from a comparison between AlexNet and ResNet-18 on the so-called 'most challenging dataset.' However, the two architectures differ not only in depth but also in convolutional kernels, receptive fields, pooling operations, parameter counts, and regularization strategies. Therefore, the manuscript cannot attribute their performance difference solely to network depth."*

**Response:**

This is a legitimate and important methodological objection. The reviewer is correct: AlexNet and ResNet-18 differ simultaneously along multiple architectural dimensions — depth, kernel sizes, receptive fields, skip connections, pooling strategies, and parameter count — and the observed performance difference on the PU dataset cannot be attributed to any single one of these dimensions based on the current experimental design alone.

We have two complementary responses.

First, we moderate the conclusion as requested. The revised manuscript replaces the strong claim "architectural depth does not guarantee generalization" with the more defensible claim: "the results are inconsistent with the assumption that architectural depth is a sufficient predictor of generalization under strict domain shifts." The AlexNet vs. ResNet-18 comparison is repositioned as an empirical observation that motivates further controlled investigation rather than a proven causal statement about depth.

Second, we propose a targeted additional experiment designed specifically to isolate the effect of depth while controlling for the confounding variables identified by the reviewer **[NEW EXPERIMENT REQUIRED]**. See instructions below.

**Manuscript change:** Section 4.2 conclusion revised to: *"These results are inconsistent with the assumption that greater architectural depth reliably predicts generalization under strict operating-condition separation. However, because AlexNet and ResNet-18 differ along multiple dimensions simultaneously — including kernel sizes, receptive fields, skip connections, and parameter count — the observed performance reversal cannot be attributed to depth alone. Disentangling the contribution of each factor requires controlled ablation experiments in which depth is varied while all other architectural dimensions are held constant. Such an investigation is proposed as a priority direction for future work."* The paper title is addressed in the note at the end of this letter.

**[NEW EXPERIMENT REQUIRED — R2.2]: Depth ablation study on the PU dataset**

*Objective:* Provide controlled evidence on whether increasing depth alone — with all other architectural dimensions held constant — improves or degrades generalization under strict domain shifts on the PU dataset.

*Design:* Construct a family of four CNN variants based on the standard 1D-CNN already evaluated in the paper, varying only the number of convolutional blocks (i.e., depth) while keeping kernel sizes, filter counts per block, pooling strategy, dropout rate, batch normalization, and the classification head strictly identical across variants:

- **Depth-1:** 1 convolutional block (16 filters, kernel 3×1, MaxPool stride 2) + adaptive pooling + classification head.
- **Depth-2:** 2 convolutional blocks (16→32 filters, kernel 3×1 each, MaxPool stride 2) + adaptive pooling + classification head.
- **Depth-3:** 3 convolutional blocks (16→32→64 filters, kernel 3×1 each, MaxPool stride 2) + adaptive pooling + classification head. This is equivalent to the existing 1D-CNN but with all kernels set to 3×1 (removing the current variation of 7, 5, 3 across blocks).
- **Depth-5:** 5 convolutional blocks (16→32→64→128→256 filters, kernel 3×1 each, MaxPool stride 2) + adaptive pooling + classification head.

All variants use the same training protocol (Adam, lr=3×10⁻⁴, batch size 64, 100 epochs, early stopping on 10% validation split). Evaluate on the PU dataset only under the existing similarity-bias-free protocol (8 rounds × 4 folds). Report Balanced Accuracy and Macro F1-Score with standard deviation.

*Expected outcomes:* (a) If performance degrades monotonically with depth, this provides direct evidence that depth itself is detrimental under strict domain shifts with limited training data per fold — supporting the moderated claim. (b) If performance first improves then degrades, this suggests an optimal depth regime and further qualifies the conclusion. (c) If performance is stable across depths, the dominant factor lies elsewhere (receptive field, skip connections, etc.), which also contradicts "deeper is better" and motivates the receptive field ablation in R2.3.

*Implementation note:* This experiment requires 4 variants × 8 rounds × 4 folds = 128 training runs on a single dataset. Using the existing SignalAI-Framework, only the model definition needs to change; all data loading, fold splitting, and evaluation code remain identical. Computational cost is negligible relative to the original experimental grid.

---

**Comment R2.3:** *"Similarly, the conclusion that 'a larger receptive field leads to greater robustness' remains speculative. The manuscript does not present any controlled experiments on convolutional kernel size or effective receptive field to support this claim."*

**Response:**

We agree. The receptive field interpretation was a post-hoc explanatory hypothesis offered to account for the AlexNet vs. ResNet-18 difference, and it is not supported by controlled evidence in the current experiments. We have revised the manuscript to present it explicitly as a hypothesis directed to future work, not as a finding of this study.

**Manuscript change:** All instances of "larger receptive fields confer greater robustness" have been replaced with: *"We hypothesize that the larger initial receptive field of AlexNet (kernel size 11×1) may contribute to its robustness under strict domain shifts by capturing broader temporal patterns that are less sensitive to acquisition-condition-specific signatures. However, this interpretation is not supported by controlled evidence in the current study and requires dedicated ablation experiments to be confirmed."* This language appears in Sections 4.2 and 5, with a corresponding addition to the future work directions.

**[NEW EXPERIMENT REQUIRED — R2.3]: Receptive field ablation on the PU dataset**

*Objective:* Provide controlled evidence on whether initial kernel size — and thus effective receptive field — affects generalization under strict domain shifts, independently of depth.

*Design:* Using the same Depth-3 variant defined in the R2.2 experiment as a fixed backbone (3 convolutional blocks, 16→32→64 filters, fixed depth), vary only the kernel size of the first convolutional layer across four values: k=3, k=7, k=11 (matching AlexNet's first kernel), k=64 (matching LeNet's first kernel). All subsequent layers retain kernel size 3×1. All other hyperparameters, pooling strategies, and training settings are held constant. Evaluate on the PU dataset under the same protocol (8 rounds × 4 folds).

*Expected outcome:* If Balanced Accuracy increases with initial kernel size, this provides direct evidence that receptive field is a meaningful factor, supporting the hypothesis. If the relationship is non-monotonic or absent, the hypothesis is weakened and the conclusion should be further moderated.

*Implementation note:* 4 variants × 8 rounds × 4 folds = 128 training runs. Can be run in parallel with the depth ablation (R2.2) at minimal additional cost. Together, R2.2 and R2.3 experiments form a 2×4 factorial mini-study (depth × kernel size) that substantially strengthens the paper's empirical basis and could justify a dedicated subsection in Section 4.
