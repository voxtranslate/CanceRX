# CanceRX: A Multi-scale Topology-Routed Phase-Gated Network for Joint Cancer Type and Histopathological Stage Classification

## Abstract

Multi-cancer deep learning classifiers have recently emerged as an alternative to per-organ diagnostic models, but the closest prior efforts share the same structural gap: none trains a single network that resolves both an organ-level and a fine-grained subtype/stage decision from one shared representation, and none reports a patient- or slide-disjoint evaluation protocol at merged-corpus scale, an omission that recent leakage-focused audits in this literature show inflates reported accuracy by several points once corrected. We introduce CanceRX, a single 22.58-million-parameter phase-gated attention network built on stacked Topological Phase-Gated (TPG) blocks that jointly performs coarse 8-organ classification and fine 26-class subtype/stage classification from one shared trunk. The trunk is built on two new operators: Cosine-Modulated Phase Attention (CMPA), which gates dot-product attention logits by the cosine of a learned pairwise phase difference, and dense Topological Pathway Routing (TPR), a four-expert Dense Mixture-of-Experts layer that is zero-initialized to provably begin training at the functional identity of the vanilla operator, eliminating by construction the auxiliary load-balancing and entropy losses that sparse top-k MoE designs require to avoid expert collapse. CanceRX is trained end-to-end on a merged, patient-/slide-disjoint corpus of 76,474 images (55,305 distinct patient or slide groups, partitioned 70/15/15 with zero group leakage) spanning brain MRI, breast and oral histopathology, cervical and leukemia cytology, colon and lung histopathology, and kidney CT. We additionally prove, in closed form, that the released wide-phase parameterization is functionally equivalent to a narrower one, exposing a concrete, zero-accuracy-cost saving of 1,034,208 parameters (4.6% of the model), and we audit every prediction through an 11-operator explainability suite with quantitative cross-method consensus, exceeding the one-to-three-method suites typical of prior work in this domain. On a genuinely held-out, patient-disjoint partition of 11,444 images, CanceRX reaches 100.00% organ-level and 99.96% stage-level accuracy, with 100.00% agreement between its two classification heads, an exact 1.0000/1.0000 consistency gap between them, and 0.0000 cross-organ leakage across all eight organs, with the small residual stage-level error confined entirely to biologically adjacent boundaries in Oral, Leukemia and Cervical. Taken together, these results establish CanceRX as the first multi-cancer classifier to combine single-trunk hierarchical decision-making, provably leakage-free evaluation at merged-corpus scale, and a closed-form efficiency guarantee within one architecture. All code is publicly released to facilitate independent verification and reuse at: [https://github.com/voxtranslate/CanceRX](https://github.com/voxtranslate/CanceRX)

---

<p align="center">
  <img src="assets/Datasets links.png" alt="Table 1: Source corpora merged into the CanceRX training set" width="900">
</p>

<p align="center">
  <em><strong>Table 1:</strong> Source corpora merged into the CanceRX training set. Modality abbreviations: H&amp;E = Hematoxylin and Eosin histopathology; Pap = Papanicolaou-stained cytology; CT = non-contrast computed tomography; MRI = T1-weighted magnetic resonance imaging.</em>
</p>

---

## Keywords

Multi-cancer classification · Phase-gated attention · Mixture-of-experts · Hierarchical classification · Histopathology · Patient-aware evaluation · Explainable AI
