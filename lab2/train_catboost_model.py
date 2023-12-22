from sklearn.model_selection import train_test_split
from ast import literal_eval
from catboost import CatBoostClassifier, Pool
import pandas as pd
import json
from time import time

###################################

DATASET_PATH = './frames_embeddings_4labels.csv'
OUTPUT_DIR = './'
SAVED_MODEL_NAME = 'frame_classifier_model'
METRICS_FILE = 'evl_metrics.json'
EMBEDDINGS_SIZE = 1536
X_COLS = [f"x{i} "for i in range(EMBEDDINGS_SIZE)]
Y_COL = 'labels'

###################################
print("LOAD DATASET")

print("read csv...",end='')
s_time = time()
df = pd.read_csv(DATASET_PATH, sep=';')
e_time = time()
print(f"ready [{round(e_time - s_time,3)} sec] !")

LABELS2ID = {label:i for i,label in enumerate(df[Y_COL].unique().tolist())}

print("Xy convertion...", sep='')
s_time = time()
X, y = df[X_COLS].to_numpy(), list(map(lambda label: LABELS2ID[label], df[Y_COL].to_list()))
e_time = time()
print(f"ready [{round(e_time - s_time,3)} sec] !")


###################################
print("SPLIT DATASET")

s_time=time()
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, 
                                                  random_state=0, stratify=y)
e_time=time()
print(f"ready [{round(e_time-s_time,3)} sec]!")

print(f"train: {len(X_train)}, {len(y_train)}")
print(f"eval: {len(X_val)}, {len(y_val)}")

###################################
print("START TRAINING")

clf = CatBoostClassifier(
    eval_metric='Accuracy',
    iterations=200,
    learning_rate=0.5,
    early_stopping_rounds=20,
    random_seed=63,
    loss_function='MultiClass',
    task_type="GPU",
    devices='0:1'
)


clf.fit(X_train, y_train,
        eval_set=(X_val, y_val), 
        verbose=True)

print('CatBoost model is fitted: ' + str(clf.is_fitted()))
print('CatBoost model parameters:')
print(clf.get_params())

###################################
print("SAVE TRAINED MODEL")

clf.save_model(f"{OUTPUT_DIR}{SAVED_MODEL_NAME}")

###################################
print("CALCULATE METRICS")

metrics = clf.eval_metrics(data=Pool(X_val, y_val),
                 metrics=['Accuracy','AUC', 'F1'])

print(metrics)

###################################
print("SAVE METRICS")

json_object = json.dumps(metrics, indent=2, ensure_ascii=False)
with open(f"{OUTPUT_DIR}{METRICS_FILE}", 'w') as fd:
    fd.write(json_object)
