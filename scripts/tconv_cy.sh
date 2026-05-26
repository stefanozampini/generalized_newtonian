# Run tests for Carreau-Yasuda convergence table
script="driver.py"
np=8
mms="-meshtype disk -disk_ref_level 5 -mms 4 -u_deg 3 -vector 1"

a1=("1")
n1=("0.2" "0.33" "0.66" "0.9" "1.1" "2")

a2=("3")
n2=("0.2" "0.33" "0.66" "1.1" "3" "4")

n3=("2.5")
a3=("0.25" "0.5" "1" "1.5" "2" "4")

n4=("4")
a4=("0.25" "0.5" "1" "1.5" "2" "4")

run_test() {
    tn=$1
    local -n aa
    local -n nn

    if [[ "$tn" == "1" ]]; then
      aa=a1
      nn=n1
    elif [[ "$tn" == "2" ]]; then
      aa=a2
      nn=n2
    elif [[ "$tn" == "3" ]]; then
      aa=a3
      nn=n3
    else
      aa=a4
      nn=n4
    fi
    for f in 0 1; do
    for a in "${aa[@]}"; do
    for n in "${nn[@]}"; do
       cy="-muinf 0 -mu0 1.0 -lambda 1.0"
       cyopts="-prefix_push carreau_yasuda_ -n $n -a $a $cy -prefix_pop"
       s=50
       dt=1
       log="./scripts/data/tconv/carreau_yasuda_n${n}_a${a}_dt${dt}_sp0_full${f}.pkl"
       out="./scripts/data/tconv/carreau_yasuda_n${n}_a${a}_dt${dt}_sp0_full${f}.out"
       cmd="mpiexec -n $np python $script $mms -max_steps $s -model carreau_yasuda $cyopts -full_step $f -split 0 -dt $dt -fstress 0 -logfile $log"
       #echo "running: $cmd | tee $out"
       $cmd | tee $out
    done
    done
    done
}

run_test 1
run_test 2
run_test 3
run_test 4
