:: This script performs local training for a TensorFlow model.

rem Training local ML model

set DATASET=cifar10
set JOB_NAME=1
set MODEL_DIR=/tmp/trained_models/%DATASET%_%JOB_NAME%
set PACKAGE_PATH=./trainer

set TRAIN_STEPS=1000
set EVAL_STEPS=100

gcloud ai-platform local train ^
        --module-name=trainer.task ^
        --package-path=%PACKAGE_PATH% ^
        --job-dir=%MODEL_DIR% ^
        -- ^
        --train-steps=%TRAIN_STEPS% ^
        --eval-steps=%EVAL_STEPS%