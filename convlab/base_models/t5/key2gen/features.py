import datasets

FEATURES = {
    "nlg": {
        "context+knowledge": datasets.Value("string"),
        "response": datasets.Value("string"),
        "knowledge": {
            "categorical": datasets.Sequence({
                "intent": datasets.Value("string"),
                "domain": datasets.Value("string"),
                "slot": datasets.Value("string"),
                "value": datasets.Value("string"),
            }), 
            "non-categorical": datasets.Sequence({
                "intent": datasets.Value("string"),
                "domain": datasets.Value("string"),
                "slot": datasets.Value("string"),
                "value": datasets.Value("string"),
            }), 
            "binary": datasets.Sequence({
                "intent": datasets.Value("string"),
                "domain": datasets.Value("string"),
                "slot": datasets.Value("string"),
            })
        }},
    "kvret": {
        "context+knowledge": datasets.Value("string"),
        "response": datasets.Value("string"),
        "knowledge": {
            "schedule": datasets.Sequence({
                "entity": datasets.Value("string"),
                "time": datasets.Value("string"),
                "date": datasets.Value("string"),
                "party": datasets.Value("string"),
                "room": datasets.Value("string"),
                "agenda": datasets.Value("string")
            }),
            "weather": datasets.Sequence({
                "entity": datasets.Value("string"),
                "today": datasets.Value("string"),
                "monday": datasets.Value("string"),
                "tuesday": datasets.Value("string"),
                "wednesday": datasets.Value("string"),
                "thursday": datasets.Value("string"),
                "friday": datasets.Value("string"),
                "saturday": datasets.Value("string"),
                "sunday": datasets.Value("string"),
            }),
            "navigate": datasets.Sequence({
                "entity": datasets.Value("string"),
                "traffic_info": datasets.Value("string"),
                "poi_type": datasets.Value("string"),
                "address": datasets.Value("string"),
                "distance": datasets.Value("string")
            })
        }},
    "opendialkg": {
        "context+knowledge": datasets.Value("string"),
        "response": datasets.Value("string"),
        "knowledge": datasets.Sequence(datasets.Sequence(datasets.Value("string"))),
        },
    "wow": {
        "context+knowledge": datasets.Value("string"),
        "response": datasets.Value("string"),
        "knowledge": datasets.Sequence(datasets.Value("string")),
        },
    "personachat": {
        "context+knowledge": datasets.Value("string"),
        "response": datasets.Value("string"),
        "knowledge": datasets.Sequence(datasets.Value("string")),
    }
}