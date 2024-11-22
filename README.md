# AVQACL: A Novel Benchmark for Audio-Visual Question Answering Continual Learning

In this paper, a novel benchmark for audio-visual question answering continual learning (AVQACL) is introduced, aiming to study fine-grained scene understanding
and spatial-temporal reasoning in videos under a continual learning setting. 

[//]: # (and propose a method <b>AV-CIL</b>. [[paper]&#40;https://arxiv.org/pdf/2308.11073.pdf&#41;])

[//]: # (<div align="center">)

[//]: # (  <img width="100%" alt="AV-CIL" src="images/model.jpg">)

[//]: # (</div>)

[//]: # ()

## Environment

Python 3.12.4

Pytorch 2.4.0

To setup the environment, please run

```
pip install -r requirements.txt
```

## Datasets

### Split-AVQA and Split-MUSIC-AVQA


The feature data extracted by the audio and visual encoders from the two datasets can be downloaded 
from the google drive [link](https://drive.google.com/open?id=1FjKwe79e0u96vdjIVwfRQ1V6SoDHe7kK). After downloading, place the 'features' folder in the current directory 
to run experiments with our proposed method.

