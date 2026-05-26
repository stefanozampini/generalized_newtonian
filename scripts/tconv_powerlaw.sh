# Run tests for powerlaw convergence figures

script="driver.py"
for vector in 0 1; do
for f in 0 1; do
for split in 0 1; do

if [[ "$split" == "1" && "$f" == "1" ]]; then
   continue
fi

if [[ "$vector" == "1" ]]; then
mms="-meshtype disk -disk_ref_level 5 -mms 4 -u_deg 3 -vector 1"
np=8
else
mms="-llx -1 -lly -1 -nx 32 -vector 0 -u_deg 2 -mms 3 -quad"
np=4
fi

for p in 1.067 1.1 1.2 1.334 1.5 3 4 5 6; do

s=200
dt=0.25
log="./scripts/data/tconv/powerlaw_p${p}_v${vector}_dt${dt}_sp${split}_full${f}.pkl"
out="./scripts/data/tconv/powerlaw_p${p}_v${vector}_dt${dt}_sp${split}_full${f}.out"
cmd="mpiexec -n $np python $script $mms -max_steps $s -powerlaw_p $p -full_step $f -split $split -dt $dt -fstress -logfile $log"
#echo "running: $cmd"
$cmd | tee $out

s=100
dt=0.5
log="./scripts/data/tconv/powerlaw_p${p}_v${vector}_dt${dt}_sp${split}_full${f}.pkl"
out="./scripts/data/tconv/powerlaw_p${p}_v${vector}_dt${dt}_sp${split}_full${f}.out"
cmd="mpiexec -n $np python $script $mms -max_steps $s -powerlaw_p $p -full_step $f -split $split -dt $dt -fstress -logfile $log"
#echo "running: $cmd"
$cmd | tee $out

s=50
dt=1
log="./scripts/data/tconv/powerlaw_p${p}_v${vector}_dt${dt}_sp${split}_full${f}.pkl"
out="./scripts/data/tconv/powerlaw_p${p}_v${vector}_dt${dt}_sp${split}_full${f}.out"
cmd="mpiexec -n $np python $script $mms -max_steps $s -powerlaw_p $p -full_step $f -split $split -dt $dt -fstress -logfile $log"
#echo "running: $cmd"
$cmd | tee $out

done

done
done
done
