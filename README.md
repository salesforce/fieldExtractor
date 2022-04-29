# Field Extraction from Forms

## Introduction

This repository contains code of [Field Extraction from Forms with Unlabeled Data](https://arxiv.org/pdf/2110.04282.pdf).

## Environment
```angular2
CUDA="11.0"
CUDNN="8"
UBUNTU="18.04"
```

## Install
~~~bash
bash install.sh
# under our project root folder
python setup.py develop 
~~~

## Data Preparation

*We have pre-processed [INV-CDIP](https://github.com/salesforce/inv-cdip) test set under datasets/.

## Reproduce Our Results
*[Download](https://console.cloud.google.com/storage/browser/sfr-field-extractor-research) our model pre-trained using [INV-CDIP](https://github.com/salesforce/inv-cdip) unlabeled train set.

```angular2
python main.py \
--model_name_or_path pretrained_model_acl2022 \
--output_dir $OUTPUT_PATH
```
## Visualization

*Download images of [INV-CDIP](https://github.com/salesforce/inv-cdip) test set and put under datasets/imgs.

```angular2
python vis_results.py --pred_path $OUTPUT_PATH/prediction_pairs.pkl
```

## Citation
If you find this codebase useful, please cite our paper:

``` latex
@article{gao2021field,
  title={Field Extraction from Forms with Unlabeled Data},
  author={Gao, Mingfei and Chen, Zeyuan and Naik, Nikhil and Hashimoto, Kazuma and Xiong, Caiming and Xu, Ran},
  journal={ACL Spa-NLP Workshop},
  year={2022}
}
```

## License
Our code and pre-trained model are released under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).

The [INV-CDIP dataset](https://github.com/salesforce/inv-cdip) is released under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). 
The underlying documents to which the dataset refers are from the [Tobacco Collections of Industry Documents Library](https://www.industrydocuments.ucsf.edu/). 
Please see [Copyright and Fair Use](https://www.industrydocuments.ucsf.edu/help/copyright/) for more information.

## Contact
Please send an email to mingfei.gao@salesforce.com if you have questions.