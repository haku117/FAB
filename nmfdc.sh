MODE="pipefsb2"  # async fsb fab"

#DATADIR=/home/gzhao/mnil/FAB/data
#DATADIR=/Users/guoyizhao/Documents/GitHub/data
#DATADIR=/Users/haku/PycharmProjects/KMgen/
DATADIR=/tmp/gz/data
RESBASEDIR=/tmp/gz/result/
SCRBASEDIR=score/
LOGBASEDIR=/tmp/gz/log/

ALG=nmf
PARAM=1000 # nnx
DIM=200,1000,2000 #rank, nnx, nny
RANK=200
#PARAM=100 # nnx
#DIM=10,100,1000 #rank, nnx, nny

#ALG=mlp
#PARAM=10,15,1

YLIST=-1

DSIZE=2k
#DF=$ALG-$PARAM-$DSIZE-d.csv

BS=10000 # 1000 10000
LR=0.00001 # 0.01
ITER=5
TIME=30

function set_dir(){
  export RESDIR=$RESBASEDIR/$PARAM-$DSIZE/$BS-$LR
  export SCRDIR=$SCRBASEDIR/$PARAM-$DSIZE/$BS-$LR
  export LOGDIR=$LOGBASEDIR/$PARAM-$DSIZE/$BS-$LR
  mkdir -p $RESDIR 
  mkdir -p $SCRDIR
  mkdir -p $LOGDIR
}

set_dir
#m=sync
i=6
#KK=10

for m in $MODE; do for i in 1 2 4;
#do for BS in 100000 200000 400000;

do echo $i-$m-$ALG-$BS -- $(date);
	set_dir
	#mpirun --mca btl_tcp_if_include 192.168.0.0/24 -n $((i+1)) --hostfile myhosts --map-by node ../test/mpitest
	mpirun --mca btl_tcp_if_include 192.168.0.0/24 -n $((i+1))  src/main/main $m $ALG $DIM $DATADIR/$ALG-$PARAM-$DSIZE.csv $RESDIR/$m-$i-$RANK.csv -1 $YLIST -1 $LR $BS $ITER $TIME 100 0.5 1 --v=2 > $LOGDIR/$m-$i-$RANK;

echo pp $RESDIR/$m-$i-$RANK -- $(date);
#src/main/postprocess $ALG $DIM $RESDIR/$m-$i-$RANK.csv $DATADIR/$ALG-$PARAM-$DSIZE.csv -1 $YLIST "-" $SCRDIR/$m-$i-$RANK.txt 0 0 &

done
done
