#!/bin/bash
IN=$1
IN_DATA=$2
OUT=$3

mkdir -p $OUT
cp "$IN/database/test.db" "$OUT/ontology.db"
cp "$IN_DATA/ontology_test.db" "$OUT/ontology.json"
cp "$IN/pytorch_model.bin" "$OUT/pytorch_model.bin"
cp "$IN/config.json" "$OUT/config.json"
