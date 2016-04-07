#!/bin/bash

for d in `seq 1 20` ; do ./traintest.sh sentiment $d ; done

