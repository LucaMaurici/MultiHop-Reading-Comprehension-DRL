# MultiHop Reading Comprehension-DRL

Reference paper: https://arxiv.org/pdf/1905.09438.pdf

## 1 Introduction
This project was centred around the topic of Reading Comprehension (RC) in the context of Question
Answering (QA); or rather accurately identifying the relevant text from several context documents to
answer a given question. More specifically, this work concerned the reference paper “Multi-hop Reading
Comprehension via Deep Reinforcement Learning based Document Traversal” by Long et al. (Long,

2019). From said paper, it may be deduced that RC is a sequential process, particularly in the case of multi-
hop. Therefore, to reach an answer, a series of independent stages are required. Firstly, starting from a

collection of documents, graphs of sentences are constructed. The second stage involves an extractor
which, through the graph traversal, identifies the appropriate section of knowledge among the various
documents. Finally, the output from the extractor is computed by a reader. More specifically it is the
Reinforced Mnemonic Reader, which had been shown to perform well compared to previous attentive
readers, as seen by Hu et al. (Hu, 2018).
As an alternative to PPO optimisation, two other possible novel approaches are proposed: Shortest Path
Policy Optimisation (ShPaPO) and Breath Visit Method.
The implementation is described below in terms of said three stages, the report then moves on to outline
the research and experiments carried out along with the respective results.

### ...

### To know more, read the file called "Project report.pdf".
