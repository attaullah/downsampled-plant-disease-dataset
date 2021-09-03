# Downsampled Plant Disease Dataset

# Introduction
[PlantVillage Disease Classification Challenge](https://www.crowdai.org/challenges/plantvillage-disease-classification-challenge)

[arxiv](https://arxiv.org/abs/1511.08060)
14 species.
>Apple, Blueberry, Cherry, Corn, Grape, Orange, Peach, Bell Pepper, Potato, Raspberry, Soybean, Squash, Strawberry, and Tomato.

The names of 38 classes are:

1. Apple Scab
2. Apple Black Rot
3. Apple Cedar Rust
4. Apple healthy
5. Blueberry healthy
6. Cherry healthy
7. Cherry Powdery Mildew
8. Corn Gray Leaf Spot
9. Corn Common Rust
10. Corn healthy
11. Corn Northern Leaf Blight
12. Grape Black Rot
13. Grape Black Measles
14. Grape Leaf Blight
15. Grape healthy
16. Orange Huanglongbing
17. Peach Bacterial Spot
18. Peach healthy
19. Bell Pepper Bacterial Spot
20. Bell Pepper healthy
21. Potato Early Blight
22. Potato healthy
23. Potato Late Blight
24. Raspberry healthy
25. Soybean healthy
26. Squash Powdery Mildew
27. Strawberry Healthy
28. Strawberry Leaf Scorch
29. Tomato Bacterial Spot
30. Tomato Early Blight
31. Tomato Late Blight
32. Tomato Leaf Mold
33. Tomato Septoria Leaf Spot
34. Tomato Two Spotted Spider Mite
35. Tomato Target Spot
36. Tomato Mosaic Virus
37. Tomato Yellow Leaf Curl Virus
38. Tomato healthy

Background class is omitted. Dataset can be downloaded from [link](https://data.mendeley.com/datasets/tywbtsjrjv/1).


# Downsampled variants
We have downsampled to `32x32` , `64x64` and `96x96`. All three downsampled variants are used in [3,4]. Downsmampled version can be generated by using script provided in `downsample_script.py` from original files.
They can be downloaded 
from [releases](https://github.com/attaullah/downsampled-plant-disease-dataset/releases)

# Citation information
original paper
```
@article{geetharamani2019identification,
  title={Identification of plant leaf diseases using a nine-layer deep convolutional neural network},
  author={Geetharamani, G and Pandian, Arun},
  journal={Computers \& Electrical Engineering},
  volume={76},
  pages={323--338},
  year={2019},
  publisher={Elsevier}
}
```
if you use downsampled variants, kindly cite our paper.
```
@misc{sahito2021better,
      title={Better Self-training for Image Classification through Self-supervision}, 
      author={Attaullah Sahito and Eibe Frank and Bernhard Pfahringer},
      year={2021},
      eprint={2109.00778},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

# References
1. J, ARUN PANDIAN; GOPAL, GEETHARAMANI (2019), “Data for: Identification of Plant Leaf Diseases Using a 9-layer Deep Convolutional Neural Network”, Mendeley Data, V1, doi: [10.17632/tywbtsjrjv.1](https://www.sciencedirect.com/science/article/abs/pii/S0045790619300023?via%3Dihub#!)
2. David. P. Hughes, Marcel Salathe (2016), An open access repository of images on plant health to enable the development of mobile disease diagnostics", [arxiv:1511.08060](https://arxiv.org/abs/1511.08060)
3. Sahito A., Frank E., Pfahringer B. (2020) Transfer of Pretrained Model Weights Substantially Improves Semi-supervised Image Classification. In: Gallagher M., Moustafa N., Lakshika E. (eds) AI 2020: Advances in Artificial Intelligence. AI 2020 . Lecture Notes in Computer Science, vol 12576. Springer, Cham. [DOI:978-3-030-64984-5_34](https://doi.org/10.1007/978-3-030-64984-5_34)
4. Sahito A., Frank E., Pfahringer B. (2021) Better Self-training for Image Classification through Self-supervision. 	[arXiv:2109.00778](https://arxiv.org/abs/2109.00778)
 
 # LICENSE
 [CC0 1.0](LICENSE)
