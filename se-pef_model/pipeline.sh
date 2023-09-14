DATASET_FOLDER='../../SE-PEF'
DATA_FOLDER='../../SE-PEF/SE-PEF'
OUTPUT_FOLDER='../../SE-PEF/SE-PEF'
SEED=42 

SAVED_MODEL=$DATASET_FOLDER'/model_10.pt'
OUTPUT_FOLDER='../created_data'
BERT='distilbert-base-uncased'
EMB_DIM=768
BATCH=32
python3 create_answer_embeddings.py --data_folder $DATA_FOLDER --embedding_dim $EMB_DIM --bert_name $BERT --batch_size $BATCH --seed $SEED --saved_model $SAVED_MODEL --output_folder $OUTPUT_FOLDER

BERT='nreimers/MiniLM-L6-H384-uncased'
EMB_DIM=384
BATCH=64
python3 create_answer_embeddings.py --data_folder $DATA_FOLDER --embedding_dim $EMB_DIM --bert_name $BERT --batch_size $BATCH --seed $SEED --output_folder $OUTPUT_FOLDER

SPLIT='val'
BERT_NAME='distilbert-base-uncased'
MODEL_PATH='../../SE-PEF/model_10.pt'
python3 testing_reranker.py --data_folder $DATA_FOLDER --bert_name $BERT_NAME --model_path $MODEL_PATH --split $SPLIT --output_folder $OUTPUT_FOLDER
BERT_NAME='sentence-transformers/all-MiniLM-L6-v2'
python3 testing_reranker.py --data_folder $DATA_FOLDER --bert_name $BERT_NAME --split $SPLIT --output_folder $OUTPUT_FOLDER

python3 testing_tag_based.py --dataset_folder $DATASET_FOLDER --data_folder $DATA_FOLDER --split $SPLIT

SPLIT='test'
BERT_NAME='distilbert-base-uncased'
MODEL_PATH='../../SE-PEF/model_10.pt'
python3 testing_reranker.py --data_folder $DATA_FOLDER --bert_name $BERT_NAME --model_path $MODEL_PATH --split $SPLIT --output_folder $OUTPUT_FOLDER
BERT_NAME='sentence-transformers/all-MiniLM-L6-v2'
python3 testing_reranker.py --data_folder $DATA_FOLDER --bert_name $BERT_NAME --split $SPLIT --output_folder $OUTPUT_FOLDER

python3 testing_tag_based.py --dataset_folder $DATASET_FOLDER --data_folder $DATA_FOLDER --split $SPLIT

SPLIT='val'
python3 fuse.py --data_folder $DATA_FOLDER --split $SPLIT 

# now change the parameters in fuse.py file and run the next line (we put them for the baseline already)

SPLIT='test'
python3 fuse.py --data_folder $DATA_FOLDER --split $SPLIT 
