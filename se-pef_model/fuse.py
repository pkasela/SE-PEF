from os.path import join

import click
from dataloader.utils import load_query_data_test
from ranx import Qrels, Run, compare, fuse, optimize_fusion

@click.command()
@click.option(
    "--data_folder",
    type=str,
    required=True
)
@click.option(
    "--split",
    type=str,
    required=True
)
def main(data_folder, split):
    test_queries = load_query_data_test(join(data_folder, f'{split}/data.jsonl'))

    expert_qrel = {q: {str(exp_id) : 1 for exp_id in test_queries[q]['expert_ids']} for q in test_queries}
    expert_qrel = Qrels(expert_qrel)


    tag_run = Run.from_file(join(data_folder, f'{split}/run_tag.json'))
    tag_run.name = 'TAG'
    print('Read Tag Run')

    bert_run = Run.from_file(join(data_folder, f'{split}/run_ranker_0.json'))
    bert_run.name = 'BERT'
    print('Read BERT Run')

    distilbert_run = Run.from_file(join(data_folder, f'{split}/run_ranker_10.json'))
    distilbert_run.name = 'DISTILBERT'
    print('Read DISTILBERT Run')


    bm25_run = Run.from_file(join(data_folder, f'{split}/bm25_user_run.json'))
    bm25_run.name = 'BM25'
    print('Read BM25 Run')
    if split == 'val':
        best_params = optimize_fusion(
            qrels=expert_qrel,
            runs=[bert_run, tag_run],
            norm="min-max",
            method="wsum",
            metric="recall@5",  # The metric to maximize during optimization
            return_optimization_report=True,
        )
        print("BERT_TAG")
        print(best_params)

        best_params = optimize_fusion(
            qrels=expert_qrel,
            runs=[distilbert_run, tag_run],
            norm="min-max",
            method="wsum",
            metric="recall@5",  # The metric to maximize during optimization
            return_optimization_report=True,
        )
        print("DISTILBERT_TAG")
        print(best_params)
        
        
        best_params = optimize_fusion(
            qrels=expert_qrel,
            runs=[bm25_run, tag_run],
            norm="min-max",
            method="wsum",
            metric="recall@5",  # The metric to maximize during optimization
            return_optimization_report=True
        )
        print("BM25_TAG")
        print(best_params)

        best_params = optimize_fusion(
            qrels=expert_qrel,
            runs=[bm25_run, bert_run],
            norm="min-max",
            method="wsum",
            metric="recall@5",  # The metric to maximize during optimization
            return_optimization_report=True
        )
        print("BM25_BERT")
        print(best_params)
        
        best_params = optimize_fusion(
            qrels=expert_qrel,
            runs=[bm25_run, distilbert_run],
            norm="min-max",
            method="wsum",
            metric="recall@5",  # The metric to maximize during optimization
            return_optimization_report=True
        )
        print("BM25_DISTILBERT")
        print(best_params)

        best_params = optimize_fusion(
            qrels=expert_qrel,
            runs=[bm25_run, bert_run, tag_run],
            norm="min-max",
            method="wsum",
            metric="recall@5",  # The metric to maximize during optimization
            return_optimization_report=True
        )
        print("BM25_BERT_TAG")
        print(best_params)
        
        best_params = optimize_fusion(
            qrels=expert_qrel,
            runs=[bm25_run, distilbert_run, tag_run],
            norm="min-max",
            method="wsum",
            metric="recall@5",  # The metric to maximize during optimization
            return_optimization_report=True
        )
        print("BM25_DISTILBERT_TAG")
        print(best_params)

    if split == 'test':
        # all_best_params = {'weights': (.0,0.5,.5)}
        # all_combined_test_run = fuse(
        #     runs=[bm25_run, distilbert_run, tag_run],  
        #     norm="min-max",       
        #     method="wsum",        
        #     params=all_best_params,
        # )
        # all_combined_test_run.name = 'BM25 + DISTILBERT + TAG'

        # all_2_best_params = {'weights': (.4,.0,.6)}
        # all_2_combined_test_run = fuse(
        #     runs=[bm25_run, distilbert_run, tag_run],  
        #     norm="min-max",       
        #     method="wsum",        
        #     params=all_best_params,
        # )
        # all_2_combined_test_run.name = 'BM25 + BERT + TAG'

        bm25_distilbert_best_params = {'weights': (.1,.9)}
        bm25_distilbert_combined_test_run = fuse(
            runs=[bm25_run, distilbert_run],  
            norm="min-max",       
            method="wsum",        
            params=bm25_distilbert_best_params,
        )
        bm25_distilbert_combined_test_run.name = 'BM25 + DISTILBERT'
        
        bm25_bert_best_params = {'weights': (.8,.2)}
        bm25_bert_combined_test_run = fuse(
            runs=[bm25_run, bert_run],  
            norm="min-max",       
            method="wsum",        
            params=bm25_bert_best_params,
        )
        bm25_bert_combined_test_run.name = 'BM25 + BERT'
        
        bm25_tag_best_params = {'weights': (.4,.6)}
        bm25_tag_combined_test_run = fuse(
            runs=[bm25_run, tag_run],  
            norm="min-max",       
            method="wsum",        
            params=bm25_tag_best_params,
        )
        bm25_tag_combined_test_run.name = 'BM25 + TAG'

        bert_tag_best_params = {'weights': (.4,.6)}
        bert_tag_combined_test_run = fuse(
            runs=[bert_run, tag_run],  
            norm="min-max",       
            method="wsum",        
            params=bert_tag_best_params,
        )
        bert_tag_combined_test_run.name = 'BERT + TAG'
        
        distilbert_tag_best_params = {'weights': (.5,.5)}
        distilbert_tag_combined_test_run = fuse(
            runs=[distilbert_run, tag_run],  
            norm="min-max",       
            method="wsum",        
            params=distilbert_tag_best_params,
        )
        distilbert_tag_combined_test_run.name = 'DISTILBERT + TAG'


        models = [
            bm25_run,
            bm25_tag_combined_test_run, 
            distilbert_run,
            bm25_distilbert_combined_test_run,  
            distilbert_tag_combined_test_run,
            bert_run,
            bert_tag_combined_test_run,
            bm25_bert_combined_test_run,
            # all_2_combined_test_run,
            # all_combined_test_run    
        ]
        
        report = compare(
            qrels=expert_qrel,
            runs=models,
            metrics=['precision@1', 'recall@3', 'recall@5', 'mrr@5', 'hit_rate@5'],
            max_p=0.01/4  # P-value threshold
        )
        print(report)

if __name__ == '__main__':
    main()