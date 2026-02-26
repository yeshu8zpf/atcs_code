opencompass opencompass/arc.py \
    --work-dir ${EVAL_DIR}/arc/${EVAL_NAME} \

opencompass opencompass/bbh.py \
    --work-dir ${EVAL_DIR}/bbh/${EVAL_NAME} \

opencompass opencompass/ifeval.py \
    --work-dir ${EVAL_DIR}/ifeval/${EVAL_NAME} \

opencompass opencompass/mmlu.py \
    --work-dir ${EVAL_DIR}/mmlu/${EVAL_NAME} \
    