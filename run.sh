MODE="sync async fsb fab"

DATADIR=/tmp/tzhou/data
RESBASEDIR=~/Code/FSB/result/
SCRBASEDIR=~/Code/FSB/score/
LOGBASEDIR=~/Code/FSB/log/

ALG=lr
PARAM=10

#ALG=mlp
#PARAM=10,15,1

YLIST=10

DSIZE=100k
#DF=$ALG-$PARAM-$DSIZE-d.csv

BS=100 # 1000 10000
LR=0.1 # 0.01
ITER=100k
TIME=60

function set_dir(){
  export RESDIR=$RESBASEDIR/$PARAM-$DSIZE/$BS-$LR
  export SCRDIR=$SCRBASEDIR/$PARAM-$DSIZE/$BS-$LR
  export LOGDIR=$LOGBASEDIR/$PARAM-$DSIZE/$BS-$LR
  mkdir -p $RESDIR
  mkdir -p $SCRDIR
  mkdir -p $LOGDIR
}

set_dir

cd ~/Code/FSB/lr-cpp/build

for i in 1 2 4 8; do for m in $MODE; do echo $i-$m -- $(date);
mpirun -n $((i+1)) src/main/main $m $ALG $PARAM $DATADIR/$ALG-$PARAM-$DSIZE-d.csv $RESDIR/$m-$i.csv -1 $YLIST 0 $LR $BS $ITER $TIME 1000 0.5 200 --v=1 > $LOGDIR/$m-$i;
done done

for PARAM in 10,1; do echo $PARAM;
for BS in 10000 1000 100; do for LR in 0.1 0.01; do echo $BS - $LR; set_dir;
for i in 1 2 4 8; do for m in $MODE; do echo $i-$m -- $(date);
mpirun -n $((i+1)) src/main/main $m $ALG $PARAM $DATADIR/$ALG-$PARAM-$DSIZE-d.csv $RESDIR/$m-$i.csv -1 $YLIST 0 $LR $BS $ITER $TIME 1000 0.5 200 --v=2 > $LOGDIR/$m-$i;
done done
done done
done

PATH=$PATH:/home/tzhou/Code/FSB/lr-cpp/build/Release/src/main

for i in 1 2 4 8; do for m in $MODE; do echo $i-$m -- $(date);
postprocess $ALG $PARAM  $RESDIR/$m-$i.csv $DATADIR/$ALG-$PARAM-$DSIZE-d.csv -1 $YLIST $DATADIR/$ALG-$PARAM-$DSIZE-p.txt $SCRDIR/$m-$i.txt 0 0
done done



