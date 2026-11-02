# Ranker training-data synthesis pipeline (3 stages)

# 1) Generate (query, candidate-paper) pairs via ref+cluster query synthesis
nohup python3 create_dataset/query_pipeline_refcluster.py full \
    --direction "Artificial Intelligence" \
    --dataset-dir ./result/dataset_refcluster \
    --seed-concurrency 10 \
    --cluster-sim 0.8 \
    --cluster-merge-sim 0.5 \
    --no-sn --resume > ./result/query_pipeline_cs.log 2>&1 &

# 2) Ranker scoring: LLM builds a checklist and gives per-paper pointwise scores (1-5)
python create_dataset/score_ranker_llm.py

# 3) Balance the score buckets for SFT
python create_dataset/balance_data.py
