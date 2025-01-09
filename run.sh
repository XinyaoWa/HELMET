##################### Run ##########################
python eval.py --config configs_test/niah_test.yaml --device hpu --use_kv_cache --use_hpu_graphs --bf16 --sdp_on_bf16
python eval.py --config configs_test/rag_short_nq_8192.yaml --device hpu --use_kv_cache --use_hpu_graphs --bf16 --sdp_on_bf16
python eval.py --config configs_test/niah_mv_8192.yaml --device hpu --use_kv_cache --use_hpu_graphs --bf16 --sdp_on_bf16
nohup python -u eval.py --config configs_test/rag_nq_8192.yaml --device hpu --use_kv_cache --use_hpu_graphs --bf16 --sdp_on_bf16 > log/rag_nq_8192.log &
nohup python -u eval.py --config configs_test/niah_mk_8192.yaml --device hpu --use_kv_cache --use_hpu_graphs --bf16 --sdp_on_bf16 > log/niah_mk_8192.log &

##################### build docker ##########################
docker build -t helmet:latest --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy -f Dockerfile .

##################### start docker ##########################
docker run -it  \
    --runtime=habana \
    --name="helmet" \
    -v /home/jenkins/xinyao:/home/jenkins/xinyao \
    -v /old_os/xinyao/:/old_os/xinyao/ \
    -e https_proxy=$https_proxy \
    -e http_proxy=$http_proxy \
    -e TOKENIZERS_PARALLELISM=false \
    -e HABANA_VISIBLE_DEVICES=all \
    -e OMPI_MCA_btl_vader_single_copy_mechanism=none \
    -e HF_HOME=/old_os/xinyao/data/ \
    --cap-add=sys_nice \
    --ipc=host \
    -w /home/jenkins/xinyao \
    helmet:latest