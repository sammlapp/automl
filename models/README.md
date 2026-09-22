---
license: mit
---

This model is a variant of Perch V2 ([google/bird-vocalization-classifier](https://www.kaggle.com/models/google/bird-vocalization-classifier/tensorFlow2/perch_v2/)) developed by Tom Denton and company at Google

This model is in ONNX format (thanks to Justin Chuby for converesion: https://huggingface.co/justinchuby/Perch-onnx)
And the classification head has been removed (thanks to Matt Weldy)

Thus, the model is much smaller and lighter-weight than the version with the classification head, and is useful for embedding-only tasks. 
