python3 data_creation.py --dataset_folder ../../SE-PEF --train_split_time '2019-12-31 23:59:59' --test_split_time '2020-12-31 23:59:59'

# set indices.query.bool.max_clause_count in elastic.yml search to 4096
python3 get_bm25_run.py --dataset_folder ../../SE-PEF/SE-PEF --cpus 10 --index_name stack_expert_answers --ip localhost --port 9200 --mapping_path ../mapping.json --train_top_k 100 --val_top_k 100 --test_top_k 100
