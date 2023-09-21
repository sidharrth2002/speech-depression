<h1 align="center">Towards AST-LLDs for the Analysis of Depression in Speech Signals</h1>
<p align="center">Sidharrth Nagappan, Chern Hong Lim and Anuja Thimali Dharmaratne</p>

---

Recent advancements in deep learning allowed the deployment of large, multimodal models for depression analysis, but the increasing complexity of such models resulted in slow deployment times. This work proposes multi-stream audio-only models for depression analysis, that use transformer weights attended to by low-level descriptors (LLD) through an attention-weighted sum. It operates on the hypothesis that handcrafted feature sets will ameliorate extensive transformer pre-training. Extensive experimentation on the DAIC-WOZ test dataset shows that a combination of an audio spectrogram transformer (AST) and a Mel-frequency cepstral coefficient (MFCC) based convolutional neural network (AST-MFCC) produces the highest accuracy in our suite of models, but reports marginally lower macro F1 scores than both a naive AST and pure LLD-based models, suggesting that the injection of extra feature streams adds a sensibility element to models and limits false positives. However, the naive transformer-based and LLD-based models are surprisingly more effective at flagging depressed patients, although at the cost of an acceptable number of false positives. Our work suggests in totality that the addition of extra feature streams adds a distinct and controllable discriminating power to existing models and is able to assist lightweight models in low-data, audio-only settings.

<p align="center">
  <img src="https://github.com/sidharrth2002/speech-depression/assets/53941721/92e03794-9d21-4916-b034-b46c6d3bd9b1">
</p>
