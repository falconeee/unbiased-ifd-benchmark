\documentclass[review]{elsarticle}

\usepackage{lineno,hyperref}
\usepackage{xcolor}
\usepackage{amsmath}
\usepackage{booktabs}
\usepackage{multirow}
\usepackage{graphicx}
\usepackage{array}
\usepackage{float}
\modulolinenumbers[5]

\journal{Expert Systems with Applications}

%%%%%%%%%%%%%%%%%%%%%%%
%% Elsevier bibliography styles
%%%%%%%%%%%%%%%%%%%%%%%
\bibliographystyle{elsarticle-num}
%%%%%%%%%%%%%%%%%%%%%%%

\begin{document}

\begin{frontmatter}

\title{Architectural Depth Does Not Guarantee Generalization: A Similarity-Bias-Free Reassessment of Deep Learning for Vibration-Based Fault Diagnosis}

%% Autores e afiliações
\author[ufes]{Gabriel Falcone da Silva\corref{mycorrespondingauthor}}
\cortext[mycorrespondingauthor]{Corresponding author}
\ead{gabriel.silva.97@edu.ufes.br}
\author[ufes]{Matheus Santos de Moura}
\author[ufes]{Celso José Munaro}
\author[ufes]{Berilhes Borges Garcia}
\author[ufes]{Flávio Miguel Varejão}

\address[ufes]{Federal University of Espírito Santo (UFES), Vitória - ES, Brazil}

\begin{abstract}
\textcolor{red}{A dominant assumption in vibration-based Intelligent Fault Diagnosis (IFD) is that deeper Deep Learning architectures provide superior generalization.} 
\textcolor{blue}{In vibration-based Intelligent Fault Diagnosis (IFD), increasing network depth is often associated with richer hierarchical representations and improved diagnostic performance, implicitly suggesting that deeper architectures may provide better generalization.}
However, most evidence supporting this assumption is derived from experimental protocols affected by similarity bias—a systematic evaluation artifact arising from random partitioning of highly correlated signal segments. This paper investigates whether the generalization advantage of deep architectures survives the elimination of similarity bias, and whether architectural depth is a reliable predictor of diagnostic robustness under realistic evaluation conditions.

We conduct a controlled investigation of \textcolor{red}{seven} \textcolor{blue}{nine} representative Deep Learning architectures \textcolor{red}{—1D-MLP, standard Convolutional Neural Network (CNN), LeNet, AlexNet, ResNet-18, Bi-directional LSTM (BiLSTM), and three Autoencoder variants—}evaluated strictly under a similarity-bias-free protocol across four public benchmark datasets \textcolor{red}{(CWRU-12k, CWRU-48k, MFPT, and Paderborn University)}. Three central hypotheses are tested: (i) whether eliminating similarity bias produces a systematic and architecture-dependent performance degradation; (ii) whether architectural depth consistently predicts generalization capability across datasets; and (iii) which datasets retain discriminative power for benchmarking modern DL architectures once similarity bias is removed.

The results reveal three findings with direct implications for IFD research. First, the elimination of similarity bias exposes a severe and architecture-dependent generalization gap: \textcolor{red}{Balanced Accuracy drops from near-perfect (>98\%) under biased splits to near-random levels for shallow models, while deep architectures reach between 62\% and 100\% depending on dataset complexity.} \textcolor{blue}{Near-perfect (>98\%) accuracy results reported in the literature under biased splits drops to near-random levels of balanced accuracy for fully connected architectures, while convolutional and recurrent architectures reach between 62\% and 100\% depending on dataset complexity}. Second, architectural depth does not reliably predict generalization: on the  \textcolor{red}{Paderborn dataset—the most realistic benchmark—AlexNet (8 layers, 72.3\% Balanced Accuracy) consistently outperforms the structurally deeper ResNet-18 (18 layers, 62.7\%)} \textcolor{blue}{most challenging benchmark, a less deep model consistently outperforms a deeper one}, suggesting that large receptive fields confer greater robustness than depth under strict domain shifts. Third, \textcolor{red}{CWRU and MFPT} \textcolor{blue}{some} datasets show performance saturation even under unbiased conditions \textcolor{red}{(e.g., ResNet-18 achieving 100\% on MFPT)}, limiting their utility for discriminating modern architectures, whereas \textcolor{red}{the Paderborn}
\textcolor{blue}{one} dataset emerges as the \textcolor{red}{only} \textcolor{blue}{main} benchmark that meaningfully differentiates architectures under realistic generalization conditions.

These findings challenge the prevailing assumption that architectural complexity drives robustness in \textcolor{red}{IFD} \textcolor{blue}{Intelligent Fault Diagnosis}, establish new unbiased performance references for \textcolor{red}{seven} \textcolor{blue}{nine} architectures across four datasets, and provide concrete guidance for benchmark selection in future \textcolor{red}{IFD} research.
\end{abstract}

\begin{keyword}
\textcolor{red}{
Deep Learning \sep Time-Domain \sep CNN \sep MLP \sep CWRU \sep AutoEncoders}
\textcolor{blue}{Deep Learning, Intelligent Fault Diagnosis, Similarity Bias, Generalization, Convolutional Neural Networks, Benchmark Evaluation, Vibration Signals, Rotating Machinery}
\end{keyword}

\end{frontmatter}

\linenumbers

\section{Introduction}

The application of Deep Learning (DL) to vibration-based Intelligent Fault Diagnosis (IFD) has expanded rapidly over the past decade \cite{Mo2025, saeed2025deep, Ali2025106958, Harith2025271}. A central narrative in this literature is that architectural progress, from shallow \textcolor{red}{CNNs} \textcolor{blue}{Convolutional Neural Networks (CNN)} to deeper residual networks and recurrent architectures, drives systematic improvements in diagnostic generalization. This narrative is largely supported by results exceeding 98\% accuracy, reported consistently across the field regardless of dataset or architecture \cite{ZHAO2020224}.

However, this apparent consensus rests on a methodological foundation that has only recently been questioned. Varejão et al. \cite{VAREJAO2025112822} demonstrated that a large fraction of published IFD results are affected by similarity bias: when raw vibration signals are segmented into overlapping or adjacent windows and randomly split, nearly identical signal patterns appear in both training and test sets. Under these conditions, models do not learn fault-discriminative features---they memorize acquisition-specific signatures. The resulting performance estimates are inflated, non-replicable, and misleading about true generalization capability.

What remains unknown, however, is whether the architectural hierarchy that the biased literature suggests, where deeper models consistently outperform shallower ones, survives the elimination of this bias. If similarity bias inflates performance non-uniformly across architectures, then the rankings and conclusions derived from biased evaluations may be not only optimistic but fundamentally incorrect. This is the central question this paper investigates.

Specifically, this work addresses three research hypotheses. First, whether the elimination of similarity bias produces a systematic, architecture-dependent performance degradation across all evaluated models. Second, whether architectural depth remains a reliable predictor of generalization capability when models are evaluated under strict operational-condition separation. Third, which benchmark datasets retain sufficient discriminative power to meaningfully differentiate architectures once similarity bias is removed.

To test these hypotheses, we evaluate \textcolor{red}{seven} \textcolor{blue}{nine} representative DL architectures - 1D-MLP, CNN, LeNet, AlexNet, ResNet-18, BiLSTM, and three Autoencoder variants (standard, sparse, and denoising) - under the similarity-bias-free experimental framework established by Varejão et al. \cite{VAREJAO2025112822} across four public datasets: CWRU-12k, CWRU-48k, MFPT, and Paderborn University (PU). The experimental framework itself is not a contribution of this work; it is adopted strictly from \cite{VAREJAO2025112822} to ensure comparability with existing unbiased baselines. Our contribution lies entirely in the investigation: in systematically applying this framework to a broader and more architecturally diverse set of DL models, and in the empirical findings that emerge from this controlled comparison.

The results challenge the prevailing assumption that depth drives robustness. On the Paderborn dataset, the most realistic benchmark in our evaluation, AlexNet outperforms ResNet-18 by nearly 10 percentage points under the unbiased protocol, a reversal that is invisible under biased evaluation. Furthermore, CWRU and MFPT datasets exhibit performance saturation even under unbiased conditions, suggesting they are no longer informative benchmarks for discriminating modern DL architectures. Only the Paderborn dataset retains the discriminative power necessary for meaningful architectural comparison under realistic generalization conditions.

Beyond these architectural findings, our results quantify for the first time the extent of performance inflation caused by similarity bias across a representative set of DL architectures, establishing new unbiased performance references that can serve as a rigorous baseline for future IFD research.

The remainder of this paper is organized as follows. Section 2 describes the similarity bias problem and the experimental methodology adopted to eliminate it, including the datasets, evaluation metrics, and training protocol. Section 3 presents the DL architectures evaluated. Section 4 reports the results and discusses the three research hypotheses. Section 5 concludes with implications for benchmark design and directions for future work.

\section{The Similarity Bias Problem and Experimental Framework}

Similarity bias refers to a systematic evaluation error that occurs when highly similar or correlated data samples are present in both training and testing sets due to inappropriate data partitioning strategies \cite{VAREJAO2025112822}. In vibration-based fault diagnosis, this bias typically arises when raw signals are segmented into overlapping or adjacent windows and then randomly split, allowing nearly identical signal patterns to appear across different subsets. As a consequence, learning models---particularly deep architectures with high representational capacity---may exploit these similarities to memorize signal characteristics rather than learning fault-relevant and generalizable features. This leads to overly optimistic performance estimates that do not reflect real-world operating conditions, where test data are acquired under distinct and previously unseen scenarios.

\subsection{Experimental Framework and Resampling Strategy}

It is imperative to state that the experimental methodology utilized in this study to eliminate similarity bias is entirely adopted from the framework proposed by Varejão et al. \cite{VAREJAO2025112822}. This section does not propose a new data partitioning method; rather, it describes the established rigorous protocol that we apply to systematically investigate the generalization capabilities of the Deep Learning architectures selected for this study. By adopting this standardized framework, implemented via the open-source \textit{vibdata} repository \cite{Vibdata}, we ensure that our findings are fully reproducible and directly comparable with existing unbiased baselines.

Unlike traditional stratified random splits, this methodology groups samples generated under the same data acquisition conditions (e.g., specific motor loads or speeds) into the same cross-validation fold. Consequently, the training and testing sets represent completely separate operating conditions, preventing the model from memorizing specific signal signatures associated with a particular acquisition session.

The original framework \cite{VAREJAO2025112822} evaluated five public datasets. However, to maintain the integrity of the unbiased evaluation, datasets that could not be partitioned in a manner that fully eliminates similarity bias (such as IMS \cite{lee2007bearing} and UOC \cite{cao2018gear}, due to run-to-failure setups or limited condition variations) were deliberately excluded from our validation. The four datasets retained for our investigation are summarized below:

\begin{itemize}
    \item \textbf{CWRU-12k \cite{CWRU2014Bearing}:} Contains data from the Case Western Reserve University bearing center sampled at 12 kHz. It includes four classes (Normal, Inner Race, Outer Race, Ball) acquired under four different loads (0 to 3 HP). The unbiased split utilizes 8 rounds of 4-fold cross-validation, rotating the working conditions.
    \item \textbf{CWRU-48k \cite{CWRU2014Bearing}:} Contains data sampled at 48 kHz from the same facility. It includes three classes (Inner Race, Outer Race, Ball) under four different loads. The unbiased split utilizes 8 rounds of 4-fold cross-validation.
    \item \textbf{MFPT \cite{MFPT2020FaultData}:} Provided by the Society for Machinery Failure Prevention Technology. \textcolor{red}{, this dataset includes three classes (Normal, Outer Race Fault, Inner Race Fault) under varying loads. Following the protocol in \cite{VAREJAO2025112822}, we use 5 rounds of 7-fold cross-validation.} \textcolor{blue}{Although the original MFPT benchmark contains three classes (Normal, Outer Race Fault, Inner Race Fault), the unbiased evaluation framework of Varejão et al. \cite{VAREJAO2025112822} retains only two classes — Outer Race Fault and Inner Race Fault — as the Normal class was not acquired under sufficiently varied operating conditions to permit similarity-bias-free partitioning. Consequently, MFPT is treated as a binary classification problem in this study. The unbiased split uses 5 rounds of 7-fold cross-validation.}   
    
    \item \textbf{PU (Paderborn) \cite{Lessmeier2016Bearing}:} A complex dataset containing healthy bearings, artificially damaged bearings, and real damaged bearings under four varying working conditions combining speed, torque, and radial force. The classes are Normal, Inner Ring, Outer Ring, and combined Inner/Outer Ring faults. It uses 8 rounds of 4-fold cross-validation.
\end{itemize}

\subsection{Training Validation Protocol}

Considering the high computational cost associated with training deep learning models, we adopted the specific validation protocol used for the Convolutional Neural Network in the reference study \cite{VAREJAO2025112822}. Instead of a nested cross-validation (typically used for hyperparameter tuning in shallow models), this study performs an external cross-validation loop to evaluate generalization. Within each training fold, a stratified split is performed to reserve 10\% of the data for validation (convergence monitoring and early stopping), while the remaining 90\% is used for gradient updates.

\subsection{Evaluation Metrics}

Accuracy is often unsuitable for fault diagnosis datasets due to class imbalance. Therefore, to provide a comprehensive evaluation, Balanced Accuracy and Macro-averaged F1-Score (Macro F1) are employed.

\begin{itemize}
    \item \textbf{Balanced Accuracy} \cite{Zhang2018ImbalancedFD}: Defined as the arithmetic mean of the recall for each class, providing a better intuition of the true positive rate across all conditions.
    \begin{equation}
        \text{Balanced Accuracy} = \frac{1}{c}\sum_{j=1}^{c}\frac{TP_{j}}{TP_{j}+FN_{j}},
        \label{eq:balanced_acc}
    \end{equation}
    where $c$ is the number of classes, while $TP_j$ and $FN_j$ represent the True Positives and False Negatives for class $j$, respectively.

    \item \textbf{Macro F1} \cite{Sokolova2009Performance}: The harmonic mean of macro-averaged precision and recall. This metric is crucial for minimizing both false positives and false negatives in imbalanced scenarios.
    \begin{equation}
        \text{Precision}_{M} = \frac{1}{c}\sum_{j=1}^{c}\frac{TP_{j}}{TP_{j}+FP_{j}},
    \end{equation}
    \begin{equation}
        F1_{\text{macro}} = 2 \cdot \frac{\text{Precision}_{M} \cdot \text{Recall}_{M}}{\text{Precision}_{M} + \text{Recall}_{M}},
    \end{equation}
    where $FP_j$ represents the False Positives for class $j$. It is important to note that $\text{Recall}_{M}$ (Macro-averaged Recall) is mathematically equivalent to the Balanced Accuracy defined in Eq. (\ref{eq:balanced_acc}).
\end{itemize}

By adopting these metrics and the strict fold division, the reported performance reflects the true generalization ability of the deep learning models, distinct from the inflated results caused by similarity bias.

\section{Architectures Under Investigation}

To conduct a rigorous evaluation and test whether architectural depth guarantees generalization under a similarity-bias-free protocol, this work investigates a representative selection of fundamental DL architectures rather than proposing new highly complex hybrid pipelines. Specifically, we adopt the models presented in the open-source benchmark for Intelligent Fault Diagnosis (IFD) developed by Zhao et al. \cite{ZHAO2020224}. The selected architectures include Multi-Layer Perceptrons (MLP) \cite{taud2017multilayer}, standard Convolutional Neural Networks (CNN) \cite{li2021survey}, Autoencoders (AE) \cite{li2023comprehensive} along with their Sparse (SAE) \cite{ng2011sparse} and Denoising (DAE) \cite{tagawa2015structured} variants, as well as adaptations of LeNet, AlexNet, and Bi-directional LSTM (BiLSTM).

\subsection{Selected Deep Learning Models}

These models were adapted to fit the specific constraints of the unbiased experimental framework used in this study. The primary modification was performed on the input layers of all networks. While the original benchmark architectures were designed for specific fixed input lengths (e.g., 1024 points), this work adopts a time-based segmentation strategy using non-overlapping windows of one second ($T=1$s). Consequently, the input sample length $L$ is determined by the specific sampling rate of each dataset, resulting in variable input dimensions across the experimental case studies. The input dimensions of the MLPs and the input layer sizes of the CNNs, AEs, and RNNs were dynamically adjusted to match the dataset-specific signal segment length $L$ (derived from the sampling rate). This adaptation ensures that the models process the exact temporal interactions within the one-second windows defined by our unbiased protocol. The specific configurations and hyperparameters for each architecture are detailed below, and summarized in Table \ref{tab:fc_ae_architectures} for fully connected and unsupervised models, and in Table \ref{tab:cnn_architectures} for convolutional and hybrid architectures:

\begin{itemize}
    \item \textbf{Multi-Layer Perceptron (1D-MLP)}: The MLP serves as a standard baseline for classification tasks due to its simple and efficient structure. In this work, the architecture is adapted to handle high-dimensional inputs by progressively reducing dimensionality. The model's input layer dynamically adapts to the input signal length $L$, which is then mapped to a sequence of 7 fully connected hidden layers with 4096, 2048, 1024, 512, 256, 128, and 64 units respectively. To improve training stability and prevent overfitting, Batch Normalization and ReLU activation functions are applied after each hidden layer. Additionally, Dropout regularization is incorporated in the initial adaptation layers with decreasing rates ($0.7$, $0.5$, and $0.4$). The output layer consists of a fully connected layer mapping the 64 features to the class probabilities using a Softmax function.
    
    \item \textbf{Autoencoders (AE) and Variants:} The Autoencoder is an unsupervised architecture structured as an encoder-decoder network. The encoder maps the input signal $x$ to a lower-dimensional latent representation $h$, performing feature extraction, while the decoder reconstructs the original input $\hat{x}$ from this latent state. All variants evaluated in this work share a common backbone architecture adapted for high-dimensional inputs. The encoder consists of four fully connected layers reducing the dimensionality from $L \to 512 \to 256 \to 128 \to 64$ (latent dimension). Each layer is followed by Batch Normalization, ReLU activation, and Dropout ($p=0.4$ for standard AE, $p=0.2$ for variants) to prevent overfitting. The decoder mirrors this structure, progressively upsampling the latent vector back to the original input dimension. We evaluated three specific variants:

    \begin{itemize}
        \item \textbf{Standard AE:} The baseline architecture trained to minimize the reconstruction error (Mean Squared Error - MSE) between the input and the output.
        
        \item \textbf{Denoising Autoencoder (DAE):} Designed to learn robust features by reconstructing the original clean input from a corrupted version. In our implementation, additive Gaussian Noise \cite{monroy2025generalized} with a factor of $0.5$ is injected into the input signal before it is fed into the encoder during the training phase. The architecture remains identical to the Standard AE.
        
        \item \textbf{Sparse Autoencoder (SAE):} Promotes the learning of salient features by incorporating a sparsity constraint. In addition to the standard architecture, a Sigmoid activation function is applied to the latent layer to bound activations between $[0, 1]$. A Kullback-Leibler (KL) divergence term \cite{cui2025generalized} is added to the loss function to constrain the average activation of these latent neurons.
    \end{itemize}

All AE variants include a final classification head (a single linear layer) attached to the latent space for fault diagnosis.

    \item \textbf{Convolutional Neural Networks (1D-CNN):} A 1D-CNN architecture capable of handling variable input signal lengths ($L$) across different datasets without structural modification was adopted. The network consists of three convolutional blocks. The first block utilizes 16 filters with a kernel size of 7, followed by Batch Normalization and standard Max Pooling (stride 2). The second block increases the depth to 32 filters with a kernel size of 5 and similar pooling. The third block comprises 64 filters with a kernel size of 3. Crucially, the final pooling layer employs Adaptive Max Pooling to fix the output temporal dimension to 16 units regardless of the input length $L$. This mechanism generates a consistent flattened feature vector of size $1024$ ($64 \text{ channels} \times 16 \text{ time steps}$). The classification head consists of two fully connected layers: a hidden layer with 128 units followed by a Dropout layer ($p=0.3$) for regularization, and a final output layer mapping to the number of classes. ReLU activation functions are applied after every convolutional and fully connected layer (except the output).

    \item \textbf{LeNet (1D Adaptation):} The network comprises two convolutional blocks. The first block applies 6 filters with a wide kernel size of 64 (stride 4, padding 30) to capture broad spectral features, followed by standard Max Pooling ($k=2, s=2$). The second block increases the depth to 16 filters with a kernel size of 5. To ensure architectural invariance to input signal length $L$, the second pooling layer utilizes Adaptive Max Pooling, which forces the output temporal dimension to a fixed size of 12 units. This mechanism generates a consistent flattened feature vector of size $192$ ($16 \text{ channels} \times 12 \text{ time steps}$). The classification head follows the classic LeNet configuration, consisting of two fully connected hidden layers with 120 and 84 units, respectively, followed by the final output layer. ReLU activation functions are employed throughout the network.

    \item \textbf{ResNet-18 (1D Adaptation):} The network begins with an initial convolutional block (stem) comprising a kernel size of 7 (stride 2) and 64 filters, followed by Batch Normalization, ReLU activation, and a Max Pooling layer (kernel size 3, stride 2). The core architecture consists of four residual stages, each containing two Basic Blocks. Each Basic Block implements two cascaded $3 \times 1$ convolutions with Batch Normalization and ReLU, alongside a residual skip connection that adds the input identity to the output to mitigate the vanishing gradient problem. The feature depth increases progressively across the stages ($64 \to 128 \to 256 \to 512$). To ensure robustness against varying input signal lengths ($L$), the final feature map is processed by a Global Adaptive Average Pooling layer. This operation condenses the temporal dimension to a single point, resulting in a fixed 512-dimensional feature vector regardless of the input size. Finally, a single fully connected layer maps these features to the class probabilities.

    \item \textbf{AlexNet (1D Adaptation):} The network features five convolutional layers designed to extract hierarchical features from the vibration signals. The first layer employs a large kernel size of 11 (stride 4) with 64 filters to capture broad spectral patterns, followed by Max Pooling. The second layer increases the depth to 192 filters with a kernel size of 5. The third, fourth, and fifth layers utilize smaller kernels of size 3 with 384, 256, and 256 filters, respectively. Max Pooling layers are applied after the first, second, and fifth convolutional blocks. To ensure architectural invariance to the input signal length $L$, an Adaptive Average Pooling layer is applied at the end of the feature extractor, fixing the temporal output dimension to 6 units. This results in a consistent flattened feature vector of size $1536$ ($256 \text{ channels} \times 6 \text{ time steps}$). The classifier consists of three fully connected layers: two hidden layers with 1024 neurons each (optimized from the original 4096 to reduce parameters) and a final output layer. Dropout is applied before the fully connected layers to mitigate overfitting, and ReLU activations are used throughout the network.

    \item \textbf{Bi-directional LSTM (BiLSTM):} The network begins with a convolutional stem designed to reduce dimensionality and extract local features. The first layer employs a large kernel size of 64 (stride 4) with 16 filters, followed by Batch Normalization and Max Pooling ($k=4, s=4$). A second convolutional block increases the depth to 32 filters. To handle variable input signal lengths ($L$) and provide a consistent sequence for the recurrent layers, an Adaptive Max Pooling layer is applied, fixing the temporal dimension to exactly 100 time steps. The extracted sequence is then processed by a 2-layer Bi-directional LSTM with a hidden state size of 64 units. The bidirectional nature allows the model to capture dependencies from both past and future contexts. The recurrent output is activated using a Tanh function, flattened, and passed through a fully connected classification head containing a hidden layer with 512 units and Dropout ($p=0.5$) for regularization.

\end{itemize}

\begin{table}[ht]
\centering
\caption{Architecture details for Fully Connected and Unsupervised Models. $L$ denotes the input signal length derived from the sampling rate.}
\label{tab:fc_ae_architectures}
\resizebox{\columnwidth}{!}{%
\begin{tabular}{@{}l p{6cm} l@{}}
\toprule
\textbf{Model} & \textbf{Layer Sequence (Units)} & \textbf{Key Features} \\ \midrule
\textbf{1D-MLP} & 
$L \to 4096 \to 2048 \to 1024 \to 512 \to 256 \to 128 \to 64 \to N_{classes}$ & 
\begin{tabular}[c]{@{}l@{}}Progressive Dropout\\ (0.7, 0.5, 0.4)\end{tabular} \\ \midrule

\textbf{AE / DAE} & 
\textbf{Encoder:} $L \to 512 \to 256 \to 128 \to 64$ (Latent)\par
\textbf{Decoder:} $64 \to 128 \to 256 \to 512 \to L$ & 
\begin{tabular}[c]{@{}l@{}}DAE: Gaussian Noise\\ injection ($\sigma=0.5$)\end{tabular} \\ \midrule

\textbf{SAE} & 
\textit{Same structure as AE} & 
\begin{tabular}[c]{@{}l@{}}Sigmoid on Latent\\ KL-Divergence Loss\end{tabular} \\ \bottomrule
\end{tabular}%
}
\end{table}

\begin{table*}[ht]
\centering
\caption{Configuration of Convolutional and Hybrid Architectures. Note that all models utilize \textbf{Adaptive Pooling} to handle variable input lengths ($L$).}
\label{tab:cnn_architectures}
\resizebox{\textwidth}{!}{%
\begin{tabular}{@{}l l l l@{}}
\toprule
\textbf{Model} & \textbf{Feature Extractor Backbone} & \textbf{Adaptive Pooling Output} & \textbf{Classification Head (FC Layers)} \\ \midrule
\textbf{1D-CNN} & 
\begin{tabular}[c]{@{}l@{}}3 Conv Blocks:\\ 1. 16 filters ($7\times1$), MP(2)\\ 2. 32 filters ($5\times1$), MP(2)\\ 3. 64 filters ($3\times1$)\end{tabular} & 
\textbf{16 time steps} & 
\begin{tabular}[c]{@{}l@{}}Flatten ($1024$ units) $\to$ 128 $\to$ Output\\ Dropout: 0.3\end{tabular} \\ \midrule

\textbf{LeNet-5} & 
\begin{tabular}[c]{@{}l@{}}2 Conv Blocks:\\ 1. 6 filters ($64\times1$), stride 4, MP(2)\\ 2. 16 filters ($5\times1$)\end{tabular} & 
\textbf{12 time steps} & 
\begin{tabular}[c]{@{}l@{}}Flatten ($192$ units) $\to$ 120 $\to$ 84 $\to$ Output\end{tabular} \\ \midrule

\textbf{AlexNet} & 
\begin{tabular}[c]{@{}l@{}}5 Conv Layers (Filters: 64, 192, 384, 256, 256)\\ Large initial kernel ($11\times1$), stride 4\\ MP applied after layers 1, 2, and 5\end{tabular} & 
\textbf{6 time steps} & 
\begin{tabular}[c]{@{}l@{}}Flatten ($1536$ units) $\to$ 1024 $\to$ 1024 $\to$ Output\\ Dropout applied\end{tabular} \\ \midrule

\textbf{ResNet-18} & 
\begin{tabular}[c]{@{}l@{}}Stem: 64 filters ($7\times1$), stride 2\\ 4 Residual Stages (Basic Blocks):\\ Filters: $64 \to 128 \to 256 \to 512$\end{tabular} & 
\textbf{1 time step (Global Avg)} & 
\begin{tabular}[c]{@{}l@{}}Flatten ($512$ units) $\to$ Output\end{tabular} \\ \midrule

\textbf{BiLSTM} & 
\begin{tabular}[c]{@{}l@{}}Hybrid Stem:\\ 1. Conv 16 filters ($64\times1$), stride 4\\ 2. Conv 32 filters ($3\times1$)\end{tabular} & 
\textbf{100 time steps} & 
\begin{tabular}[c]{@{}l@{}}2-Layer BiLSTM ($H=64$) $\to$ Flatten\\ FC ($512$ units) $\to$ Output\\ Dropout: 0.5\end{tabular} \\ \bottomrule
\end{tabular}%
}
\end{table*}

\subsection{Training Protocol}

To ensure a fair and rigorous comparison, a standardized training protocol, summarized in Table \ref{tab:hyperparameters}, was applied across all experiments. All models were optimized using the \textbf{Adam} algorithm with a fixed learning rate of $3 \times 10^{-4}$ and a batch size of $64$. The standard training duration was set to $100$ epochs for the supervised models (MLP, CNNs, and RNNs). For the Autoencoder-based architectures (AE, SAE, and DAE), a two-stage training strategy was employed to ensure robust feature extraction: the unsupervised reconstruction phase (pre-training) was conducted for $100$ epochs, followed by an additional $100$ epochs for the supervised fine-tuning of the classification head. 

\textcolor{blue}{It should be noted that a unified training protocol was deliberately adopted across all architectures. This choice prioritizes reproducibility and reflects typical practitioner deployment conditions, consistent with the benchmark study of Zhao et al. \cite{ZHAO2020224}. It does not represent the theoretical performance ceiling of individual architectures under optimized configurations. Readers should interpret the results as a comparison of architectures under standardized conditions, rather than as a comparison of their absolute maximum capabilities. Per-architecture hyperparameter optimization under the full nested cross-validation protocol is beyond the scope of this work.}

\begin{table}[ht]
\centering
\caption{Global Training Hyperparameters applied across all experiments.}
\label{tab:hyperparameters}
\begin{tabular}{@{}l c@{}}
\toprule
\textbf{Hyperparameter} & \textbf{Value} \\ \midrule
Optimizer & Adam \\
Learning Rate & $3 \times 10^{-4}$ \\
Batch Size & 64 \\
Loss Function & Cross-Entropy (Classifiers) / MSE (Reconstruction) \\
Training Epochs & 100 (Supervised) / 100+100 (AEs) \\ \bottomrule
\end{tabular}
\end{table}

\subsection{Computational Framework}

The experimental pipeline was implemented using PyTorch and is orchestrated through the SignalAI-Framework \cite{SignalAIFramework}, a modular environment designed for rigorous signal processing evaluations. The source code developed, training scripts, and unified data for full reproduction of the experiments are publicly available in our repository at \cite{FalconeBenchmark}.

To ensure reproducibility and consistent comparisons with the state-of-the-art, the deep learning model architectures (MLP, CNN, and Autoencoders) were adapted directly from the DL-based Intelligent Diagnosis Benchmark repository \cite{ZhaoBenchmark, ZHAO2020224}. These implementations served as the baseline and were integrated into our pipeline to undergo the rigorous unbiased evaluation investigated in this work.

Furthermore, data management and standardized loading for the vibration datasets were handled by the vibdata library \cite{Vibdata, VAREJAO2025112822}. This repository provides a unified interface for 1D vibration data, ensuring that data partitioning and preprocessing steps are applied consistently across different experiments. The use of this open-source ecosystem promotes transparency and facilitates the replication of the results presented herein.

\section{Results and Discussion}

In this section, we present the experimental results obtained from the evaluation of the selected Deep Learning architectures across the four datasets: CWRU-12k, CWRU-48k, MFPT, and Paderborn (PU). Rather than merely reporting performance metrics, this analysis is explicitly structured to test the three central research hypotheses articulated in Section 1. All results are reported in terms of Balanced Accuracy and Macro F1-Score to account for class imbalances.

Table \ref{tab:all_results} summarizes the performance of all evaluated architectures under the rigorous unbiased framework, and Table \ref{tab:baseline_comparison} compares our best-performing models against the established unbiased baselines from Varejão et al. \cite{VAREJAO2025112822}.

\subsection{Hypothesis 1: Architecture-Dependent Performance Degradation}

\textbf{Hypothesis:} \textit{The elimination of similarity bias produces a systematic and architecture-dependent performance degradation across all evaluated models.}

\textbf{Data Analysis:} Under standard, randomized data splits, the literature frequently reports near-perfect accuracies (>98\%) for virtually all models \cite{ZHAO2020224}. However, as shown in Table \ref{tab:all_results}, enforcing a strict separation of operating conditions reveals a severe generalization gap. Crucially, this degradation is not uniform. Shallow and fully connected models (1D-MLP, Standard AE, SAE, DAE) experienced a catastrophic collapse in performance. On the MFPT dataset, these architectures achieved a Balanced Accuracy of exactly 50.00\% (\textcolor{red}{random guessing for the given class distribution}\textcolor{blue}{The zero standard deviation confirms that these models produced a completely degenerate prediction pattern — predicting a single class for all instances across all folds}), and on CWRU datasets, they hovered between 34\% and 43\%. Due to this complete inability to extract domain-invariant features, these models were excluded from the PU dataset experiments. In contrast, deep convolutional and recurrent models (1D-CNN, LeNet, AlexNet, ResNet-18, BiLSTM) maintained significantly higher robustness, scoring between 68\% and 100\% across the simpler datasets.

\textbf{Conclusion:} The hypothesis is confirmed. Similarity bias inflates performance asymmetrically. Shallow architectures rely almost entirely on data leakage (memorization of acquisition-specific signatures) to achieve the high accuracies reported in the biased literature. Deep architectures, while also suffering performance drops, possess intrinsic feature-extraction capabilities that provide a measurable degree of true generalization.

\begin{table*}[ht]
\centering
\caption{Consolidated experimental results comparing Balanced Accuracy (B.Acc) and Macro F1-Score (F1) across all four datasets under the unbiased evaluation. \textcolor{red}{Values are reported as decimal fractions (0-1).}
\textcolor{blue}{Values (\%) are reported as mean ± standard deviation across all cross-validation folds.}
Best results for each dataset are highlighted in \textbf{bold}. Note: MLP and AE variants were excluded from PU due to poor generalization in previous tasks.}
\label{tab:all_results}

\setlength{\tabcolsep}{3.5pt} 
\renewcommand{\arraystretch}{1.2}

\resizebox{\textwidth}{!}{%
\begin{tabular}{@{}l cc cc cc cc@{}}
\toprule
\multirow{2}{*}{\textbf{Model}} & \multicolumn{2}{c}{\textbf{CWRU-12k}} & \multicolumn{2}{c}{\textbf{CWRU-48k}} & \multicolumn{2}{c}{\textbf{MFPT}} & \multicolumn{2}{c}{\textbf{Paderborn (PU)}} \\ \cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7} \cmidrule(lr){8-9}
 & \textbf{B.Acc} & \textbf{F1} & \textbf{B.Acc} & \textbf{F1} & \textbf{B.Acc} & \textbf{F1} & \textbf{B.Acc} & \textbf{F1} \\ 
 
%\midrule
%1D-MLP & $0.4157 \pm 0.0089$ & $0.3989 \pm 0.0101$ & $0.3580 \pm 0.0121$ & $0.2905 \pm 0.0101$ & $0.5000 \pm 0.0000$ & $0.3333 \pm 0.0000$ & \multicolumn{2}{c}{-} \\
%Standard AE & $0.4266 \pm 0.0087$ & $0.4105 \pm 0.0095$ & $0.3497 \pm 0.0229$ & $0.2895 \pm 0.0196$ & $0.5000 \pm 0.0000$ & $0.3333 \pm 0.0000$ & \multicolumn{2}{c}{-} \\
%SAE & $0.4303 \pm 0.0086$ & $0.4144 \pm 0.0085$ & $0.3551 \pm 0.0111$ & $0.3133 \pm 0.0172$ & $0.5000 \pm 0.0000$ & $0.3333 \pm 0.0000$ & \multicolumn{2}{c}{-} \\
%DAE & $0.4005 \pm 0.0196$ & $0.3471 \pm 0.0213$ & $0.3493 \pm 0.0201$ & $0.2795 \pm 0.0280$ & $0.5000 \pm 0.0000$ & $0.3333 \pm 0.0000$ & \multicolumn{2}{c}{-} \\ \midrule

%1D-CNN & $0.8786 \pm 0.0128$ & $0.8749 \pm 0.0134$ & $0.8659 \pm 0.0172$ & $0.8704 \pm 0.0165$ & $0.9000 \pm 0.0278$ & $0.8761 \pm 0.0369$ & $0.7156 \pm 0.0328$ & $\mathbf{0.6892 \pm 0.0376}$ \\
%LeNet & $0.8096 \pm 0.0155$ & $0.8070 \pm 0.0153$ & $0.8107 \pm 0.0099$ & $0.8019 \pm 0.0141$ & $0.9190 \pm 0.0117$ & $0.8950 \pm 0.0120$ & $0.7154 \pm 0.0235$ & $0.6800 \pm 0.0262$ \\
%AlexNet & $0.8157 \pm 0.0254$ & $0.8128 \pm 0.0265$ & $0.7125 \pm 0.0287$ & $0.7036 \pm 0.0329$ & $0.8476 \pm 0.0190$ & $0.8083 \pm 0.0320$ & $\mathbf{0.7230 \pm 0.0355}$ & $0.6884 \pm 0.0493$ \\
%ResNet-18 & $\mathbf{0.8940 \pm 0.0215}$ & $\mathbf{0.8861 \pm 0.0240}$ & $0.8246 \pm 0.0249$ & $0.8199 \pm 0.0238$ & $\mathbf{1.0000 \pm 0.0000}$ & $\mathbf{1.0000 \pm 0.0000}$ & $0.6273 \pm 0.0308$ & $0.5788 \pm 0.0489$ \\
%BiLSTM & $0.8247 \pm 0.0158$ & $0.8193 \pm 0.0175$ & $\mathbf{0.8716 \pm 0.0204}$ & $\mathbf{0.8623 \pm 0.0234}$ & $0.9286 \pm 0.0000$ & $0.9048 \pm 0.0000$ & $0.7038 \pm 0.0350$ & $0.6621 \pm 0.0443$ \\ \bottomrule

\midrule
1D-MLP & $41.57 \pm 0.89$ & $39.89 \pm 1.01$ & $35.80 \pm 1.21$ & $29.05 \pm 1.01$ & $50.00 \pm 0.00$ & $33.33 \pm 0.00$ & \multicolumn{2}{c}{-} \\
Standard AE & $42.66 \pm 0.87$ & $41.05 \pm 0.95$ & $34.97 \pm 2.29$ & $28.95 \pm 1.96$ & $50.00 \pm 0.00$ & $33.33 \pm 0.00$ & \multicolumn{2}{c}{-} \\
SAE & $43.03 \pm 0.86$ & $41.44 \pm 0.85$ & $35.51 \pm 1.11$ & $31.33 \pm 1.72$ & $50.00 \pm 0.00$ & $33.33 \pm 0.00$ & \multicolumn{2}{c}{-} \\
DAE & $40.05 \pm 1.96$ & $34.71 \pm 2.13$ & $34.93 \pm 2.01$ & $27.95 \pm 2.80$ & $50.00 \pm 0.00$ & $33.33 \pm 0.00$ & \multicolumn{2}{c}{-} \\ \midrule

1D-CNN & $87.86 \pm 1.28$ & $87.49 \pm 1.34$ & $86.59 \pm 1.72$ & $87.04 \pm 1.65$ & $90.00 \pm 2.78$ & $87.61 \pm 3.69$ & $71.56 \pm 3.28$ & $\mathbf{68.92 \pm 3.76}$ \\
LeNet & $80.96 \pm 1.55$ & $80.70 \pm 1.53$ & $81.07 \pm 0.99$ & $80.19 \pm 1.41$ & $91.90 \pm 1.17$ & $89.50 \pm 1.20$ & $71.54 \pm 2.35$ & $68.00 \pm 2.62$ \\
AlexNet & $81.57 \pm 2.54$ & $81.28 \pm 2.65$ & $71.25 \pm 2.87$ & $70.36 \pm 3.29$ & $84.76 \pm 1.90$ & $80.83 \pm 3.20$ & $\mathbf{72.30 \pm 3.55}$ & $68.84 \pm 4.93$ \\
ResNet-18 & $\mathbf{89.40 \pm 2.15}$ & $\mathbf{88.61 \pm 2.40}$ & $82.46 \pm 2.49$ & $81.99 \pm 2.38$ & $\mathbf{100.00 \pm 0.00}$ & $\mathbf{100.00 \pm 0.00}$ & $62.73 \pm 3.08$ & $57.88 \pm 4.89$ \\
BiLSTM & $82.47 \pm 1.58$ & $81.93 \pm 1.75$ & $\mathbf{87.16 \pm 2.04}$ & $\mathbf{86.23 \pm 2.34}$ & $92.86 \pm 0.00$ & $90.48 \pm 0.00$ & $70.38 \pm 3.50$ & $66.21 \pm 4.43$ \\
\end{tabular}%
}
\end{table*}

\subsection{Hypothesis 2: Architectural Depth as a Predictor of Generalization}

\textbf{Hypothesis:} \textit{Architectural depth reliably predicts generalization capability across datasets under strict operational-condition separation.}

\textbf{Data Analysis:} The assumption that "deeper is better" holds true for the simpler benchmarks. As seen in Table \ref{tab:all_results}, the deepest convolutional model evaluated (ResNet-18) achieved state-of-the-art performance on CWRU-12k (89.40\%) and perfect generalization on MFPT (100\%). The deep recurrent architecture (BiLSTM) also dominated the high-frequency CWRU-48k dataset (87.16\%). However, the Paderborn (PU) dataset---which introduces real damage profiles and strict, complex domain shifts---presents a stark reversal of this trend. On the PU dataset, simpler CNN architectures with larger initial kernels, specifically AlexNet (72.30\%) and the standard 1D-CNN (71.56\%), substantially outperformed the deeper ResNet-18 (62.73\%). It is important to note that while ResNet-18 is structurally significantly deeper (18 parameterized layers with residual connections) compared to the shallower AlexNet (8 parameterized layers), AlexNet achieved higher robustness.

\textbf{Conclusion:} The hypothesis is rejected. Architectural depth does not guarantee generalization. When training data is limited and strictly separated by operating conditions (as in realistic scenarios), increased depth can exacerbate overfitting to the source domain. The superior performance of AlexNet on the most challenging dataset suggests that models with larger receptive fields (e.g., its initial $11 \times 1$ kernels) are more capable of extracting broad, robust spectral features under strict domain shifts than deeper architectures relying on smaller kernels like ResNet-18.

\subsection{Hypothesis 3: Discriminative Power of Benchmark Datasets}

\textbf{Hypothesis:} \textit{Certain benchmark datasets lose their discriminative power for differentiating modern DL architectures once similarity bias is removed.}

\textbf{Data Analysis:} Table \ref{tab:baseline_comparison} illustrates how our deep models performed against the traditional Machine Learning baselines established in \cite{VAREJAO2025112822}. For CWRU-12k, the performance gain of ResNet-18 over a simple DNN was marginal (+0.99\%), indicating a performance ceiling. Most notably, ResNet-18 \textcolor{red}{completely solved} \textcolor{blue}{achieved perfect Balanced Accuracy under the evaluated protocol}  the MFPT dataset under the unbiased protocol, achieving 100\% accuracy (+3.12\% over the baseline). While BiLSTM showed strong improvements on CWRU-48k (+9.52\%), the performance on CWRU and MFPT generally indicates saturation. Conversely, the PU dataset proved to be highly discriminative; it not only pushed the overall accuracy down to the 60\%--72\% range but also effectively differentiated the generalization capabilities among the deep architectures themselves (e.g., separating AlexNet from ResNet-18 by nearly 10 percentage points).

\textbf{Conclusion:} The hypothesis is confirmed. CWRU and MFPT datasets exhibit performance saturation even under unbiased conditions. Because modern deep architectures can effectively "solve" them without data leakage, these datasets offer limited utility for future IFD benchmarking. The Paderborn dataset emerges as the only benchmark evaluated that retains the complexity required to meaningfully differentiate the generalization limits of modern DL architectures.

\begin{table}[ht]
\centering
\caption{Comparison between the best method evaluated in this work and the best baseline reported in Varejão et al. \cite{VAREJAO2025112822} under the unbiased split. Values represent Balanced Accuracy.}
\label{tab:baseline_comparison}
\resizebox{\columnwidth}{!}{%
\begin{tabular}{@{}l c c c@{}}
\toprule
\textbf{Dataset} & \textbf{Reference Baseline \cite{VAREJAO2025112822}} & \textbf{Our Best Model} & \textbf{Improvement} \\ \midrule
\textbf{CWRU-12k} & $0.8841$ (DNN) & \textbf{$0.8940$} (ResNet-18) & $+0.99\%$ \\
\textbf{CWRU-48k} & $0.7764$ (RF) & \textbf{$0.8716$} (BiLSTM) & $+9.52\%$ \\
\textbf{MFPT} & $0.9688$ (NN) & \textbf{$1.0000$} (ResNet-18) & $+3.12\%$ \\
\textbf{PU} & $0.6273$ (RF) & \textbf{$0.7230$} (AlexNet) & $+9.57\%$ \\ \bottomrule
\end{tabular}%
}
\end{table}

\section{Conclusions and Future Work}

This study investigated the prevailing assumption in vibration-based Intelligent Fault Diagnosis (IFD) that deeper Deep Learning architectures inherently guarantee superior generalization. By systematically evaluating \textcolor{red}{seven} \textcolor{blue}{nine}  representative DL architectures across four public datasets under a rigorous similarity-bias-free protocol, we quantified the true generalization capabilities of these models under realistic domain shifts.

The experimental results yield three major conclusions, with profound implications for how IFD research should be benchmarked moving forward:

\begin{enumerate}
    \item \textbf{Benchmark Saturation and the Fallacy of Depth:} The most salient finding of this investigation is that the Paderborn University (PU) dataset is the only evaluated benchmark that retains the discriminative power necessary to meaningfully evaluate modern DL architectures. Simpler datasets like CWRU and MFPT exhibit performance saturation even without similarity bias (e.g., ResNet-18 \textcolor{red}{solving} \textcolor{blue}{classifying} MFPT with 100\% accuracy), rendering them inadequate for testing modern generalization limits. Crucially, on the challenging PU dataset, architectural depth failed to predict robustness: the shallower AlexNet consistently outperformed the deeper ResNet-18. This challenges the dominant "deeper is better" narrative and suggests that large receptive fields are more critical than sheer depth for handling strict domain shifts.
    
    \item \textbf{The Severity of Similarity Bias:} The near-perfect accuracies (>98\%) routinely reported in IFD literature are largely artifacts of data leakage. Eliminating this bias exposes a severe and architecture-dependent performance gap. Shallow models (1D-MLPs) and standard Autoencoders suffer a \textcolor{red}{catastrophic collapse to near-random performance} \textcolor{blue}{severe performance degradation to near-random results}  under strict condition separation, \textcolor{red} {proving that their success in biased evaluations relies entirely on memorizing} \textcolor{blue}{suggesting that their performance under biased conditions is substantially attributable to memorization of}     acquisition-specific signatures rather than learning robust, fault-discriminative features.
    
    \item \textbf{Necessity of Domain-Separated Protocols:} While deep convolutional and recurrent networks (ResNet, BiLSTM) demonstrated superior robustness compared to shallow models, they still suffered significant degradation under unbiased conditions. This underscores that architectural complexity alone cannot fully overcome domain shifts. Consequently, standard randomized data splitting is \textcolor{red}{fundamentally} \textcolor{blue}{potentially} inadequate for IFD research. Validating diagnostic models for real-world industrial applications strictly requires domain-separated experimental designs.
\end{enumerate}

Based on the limitations identified and the insights gained from this investigation, several directions for future research are proposed:

\begin{itemize}

    \item \textcolor{blue}{\textbf{Dedicated Hyperparameter Optimization:} A natural continuation of this work involves performing per-architecture hyperparameter search within a full nested cross-validation protocol. Evaluating models under dedicated configurations will complement this study by establishing their theoretical performance ceilings beyond standardized conditions.}
    \item \textbf{Evaluation of 2D Input Representations:} This work focused exclusively on time-domain (1D) signals. A critical next step is to extend this unbiased framework to evaluate models trained on two-dimensional time-frequency representations (e.g., STFT spectrograms, CWT scalograms). Investigating whether 2D representations provide invariant features that are inherently more robust to similarity bias than raw 1D signals is a promising avenue.
    
    \item \textbf{Domain Generalization (DG) and Adaptation (DA):} Since purely architectural scaling (increasing depth) failed to guarantee generalization, future work must shift toward explicit Domain Generalization and Domain Adaptation techniques. Methods designed to align feature distributions across different working conditions are necessary to bridge the remaining performance gap on complex benchmarks like the PU dataset.
    
    \item \textbf{Hybrid Architectures with Attention Mechanisms:} Incorporating attention mechanisms (e.g., Transformer-based blocks) into feature extraction pipelines could allow models to dynamically focus on the most informative temporal segments, mitigating the impact of nuisance variables and acquisition-specific noise that traditional convolutions fail to ignore.
\end{itemize}

\section*{Acknowledgment}
The authors thank for the support of: Conselho Nacional de Desenvolvimento Científico e Tecnológico – Brasil (CNPq) and Fundação de Amparo à Pesquisa e Inovação de Espírito Santo (Fapes) - TO 1138/2025, 2025-Z9VRS.

\section*{Declaration of Generative AI and AI-assisted technologies in the writing process}

During the preparation of this manuscript, the authors used ChatGPT and Claude to assist with language refinement and readability improvements. The generated outputs were critically reviewed, revised, and validated by the authors. The authors assume full responsibility for the content and conclusions presented in this publication.

\bibliography{mybibfile}

\end{document}