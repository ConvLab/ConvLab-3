set -e
for dataset_name in dailydialog metalwoz tm1 tm2 tm3 sgd multiwoz21
do
    bash get_keywords.sh ${dataset_name}
done