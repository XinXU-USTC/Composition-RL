model=/model/to/evaluate
temp=0.6
topp=0.95
topk=20
tp=1
len=30720


bash start_generate.sh \
    --model $model \
    --datasets imo_answer  \
    --temperature $temp \
    --top-p $topp \
    --top-k $topk \
    --tp $tp \
    --length $len \
    --n 4

bash start_generate.sh \
    --model $model \
    --datasets aime aime25 beyond_aime \
    --temperature $temp \
    --top-p $topp \
    --top-k $topk \
    --tp $tp \
    --length $len \
    --n 32

bash start_generate.sh \
    --model $model \
    --datasets gpqa \
    --temperature $temp \
    --top-p $topp \
    --top-k $topk \
    --tp $tp \
    --length $len \
    --n 8

bash start_generate.sh \
    --model $model \
    --datasets mmlu-pro \
    --temperature $temp \
    --top-p $topp \
    --top-k $topk \
    --tp $tp \
    --length $len \
    --n 1
