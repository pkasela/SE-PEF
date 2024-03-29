# SE-PEF 

## To create the dataset:

To create the dataset use the data provided in [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8332748.svg)](https://doi.org/10.5281/zenodo.8332748) and run the `pipeline.sh` file in `se-pef_data` folder (change the flag values to adjust file paths).
This will create first the datafiles required for training purposes and the bm25 run, omit the `get_bm25_run.py` command if you do want just to create basic dataset, the bm25 run is required only to reproduce the results of the paper.

## To create the baseline:

To create the baseline provided in the paper run the `pipeline.sh` file in `se-pef_model` folder (change the flag values to adjust file paths).
- `create_answer_embeddings.py`: Create embedding of the `answer_collection.json` file created in the the previous step in the dataset folder.
- `testing_reranker.py`: Uses the bm25 run and a dense retriever and compute score for each expert.
- `testing_tag_based.py`: Creates a personalized score using tags for each expert and each user asking question.
- `fuse.py`: when used with val split does a grid search and outputs the optimal weighted sum parameter, when used with test split computes the score (you need to manually put the optimal parameters in the file found using the validation split).


If you use this work please cite the following:

```
@inproceedings{
  10.1145/3624918.3625335,
  author = {Kasela, Pranav and Pasi, Gabriella and Perego, Raffaele},
  title = {SE-PEF: a Resource for Personalized Expert Finding},
  year = {2023},
  isbn = {9798400704086},
  publisher = {Association for Computing Machinery},
  address = {New York, NY, USA},
  url = {https://doi.org/10.1145/3624918.3625335},
  doi = {10.1145/3624918.3625335},
  booktitle = {Proceedings of the Annual International ACM SIGIR Conference on Research and Development in Information Retrieval in the Asia Pacific Region},
  pages = {288–309},
  numpages = {22},
  series = {SIGIR-AP '23}
}
```
