#!/bin/sh
# irii-soft

DIR=09-60

for i in "$DIR"/*
do
    echo "Patching" $i
    patch -p1 < $i
done

echo "Done"
