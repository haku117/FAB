MODE="sync async fsb fab"

DATADIR=/tmp/tzhou/data
RESBASEDIR=~/Code/FSB/result/
SCRBASEDIR=~/Code/FSB/score/
LOGBASEDIR=~/Code/FSB/log/

ALG=lr
PARAM=10
PARAMNAME=10

#ALG=mlp
#PARAM=10,15,1
#PARAMNAME=10,15,1

ALG=CNN
PARAM=100-20,c,5-1,a,relu-1,p,max,2-1,c,5-1,a,relu-1,p,max,2-1
PARAMNAME=100,20c5p2,1c5p2,1

YLIST=10

DSIZE=100k
#DF=$ALG-$PARAMNAME-$DSIZE-d.csv

BS=100 # 1000 10000
LR=0.1 # 0.01
ITER=100k
TIME=60

function set_dir(){
  export RESDIR=$RESBASEDIR/$PARAMNAME-$DSIZE/$BS-$LR
  export SCRDIR=$SCRBASEDIR/$PARAMNAME-$DSIZE/$BS-$LR
  export LOGDIR=$LOGBASEDIR/$PARAMNAME-$DSIZE/$BS-$LR
  mkdir -p $RESDIR
  mkdir -p $SCRDIR
  mkdir -p $LOGDIR
}

set_dir

cd ~/Code/FSB/FAB/build

for i in 1 2 4 8; do for m in $MODE; do echo $i-$m -- $(date);
mpirun -n $((i+1)) src/main/main $m $ALG $PARAM $DATADIR/$ALG-$PARAMNAME-$DSIZE-d.csv $RESDIR/$m-$i.csv -1 $YLIST 0 $LR $BS $ITER $TIME 1000 0.5 200 --v=1 > $LOGDIR/$m-$i;
done done

for PARAM in 10,1; do echo $PARAM;
for BS in 10000 1000 100; do for LR in 0.1 0.01; do echo $BS - $LR; set_dir;
for i in 1 2 4 8; do for m in $MODE; do echo $i-$m -- $(date);
mpirun -n $((i+1)) src/main/main $m $ALG $PARAM $DATADIR/$ALG-$PARAMNAME-$DSIZE-d.csv $RESDIR/$m-$i.csv -1 $YLIST 0 $LR $BS $ITER $TIME 1000 0.5 200 --v=1 > $LOGDIR/$m-$i;
done done
done done
done

PATH=$PATH:/home/tzhou/Code/FSB/FAB/build/Release/src/main

for i in 1 2 4 8; do for m in $MODE; do echo $i-$m -- $(date);
postprocess $ALG $PARAM  $RESDIR/$m-$i.csv $DATADIR/$ALG-$PARAMNAME-$DSIZE-d.csv -1 $YLIST $DATADIR/$ALG-$PARAMNAME-$DSIZE-p.txt $SCRDIR/$m-$i.txt 0 0
done done



