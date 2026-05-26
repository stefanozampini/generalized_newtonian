exe="driver.py"

# p-Laplacian
args="-atol 1.e-8 -u_deg 1 -mms 0 -max_steps 200 -adaptive_dt 1 -direct 1 -meshfile ./meshes/lshaped_withtags_tri2.msh"
for r in 0 1 2; do
for p in 1.5 3 4 8 10 20 40 80 100; do
   if [[ "$p" == "1.5" ]]; then
     dt="-dt 10"
   elif [[ "$p" == "3" ]]; then
     dt="-dt 9"
   elif [[ "$p" == "4" ]]; then
     dt="-dt 4"
   elif [[ "$p" == "8" ]]; then
     dt="-dt 0.5"
   elif [[ "$p" == "10" ]]; then
     dt="-dt 2.e-1"
   elif [[ "$p" == "20" ]]; then
     dt="-dt 1.e-1"
   elif [[ "$p" == "40" ]]; then
     dt="-dt 5.e-2"
   elif [[ "$p" == "80" ]]; then
     dt="-dt 2.0e-2"
   else
     dt="-dt 1.5e-2"
   fi
   rr=$((3 + $r))
   ref="-refine $rr"
   log="./scripts/data/compare_newton/powerlaw_p${p}_r${r}.pkl"
   out="./scripts/data/compare_newton/powerlaw_p${p}_r${r}.out"
   cmd="mpiexec -n 8 python $exe $args $dt $ref -powerlaw_p $p -logfile $log"
   #echo $cmd
   $cmd | tee $out
done
done

# optimal design
args="-atol 1.e-8 -u_deg 1 -mms 0 -max_steps 200 -direct 1 -model optimal -adaptive_dt 1 -dt 1.e0"
for r in 0 1 2; do
for l in 0 1; do
   if [[ "$l" == "0" ]]; then
     rr=$((6 + $r))
     nx=$((2**${rr}))
     lam="-nx $nx -optimal_lambda 0.0084"
   else
     rr=$((3 + $r))
     lam="-optimal_lambda 0.0145 -refine $rr -meshfile ./meshes/lshaped_withtags_tri2.msh"
   fi
   log="./scripts/data/compare_newton/optimal_l${l}_r${r}.pkl"
   out="./scripts/data/compare_newton/optimal_l${l}_r${r}.out"
   cmd="mpiexec -n 8 python $exe $args $lam -logfile $log"
   #echo $cmd
   $cmd | tee $out
done
done

# regularized Bingham
args="-atol 1.e-8 -u_deg 2 -mms 5 -max_steps 200 -direct 1 -model bingham -adaptive_dt 1 -dt 1.e0 -vector 1"
for r in 0 1 2; do
for e in 2 3 4; do
   rr=$((6 + $r))
   nx=$((2**${rr}))
   bin="-bingham_eps 1.e-${e} -nx ${nx}"
   log="./scripts/data/compare_newton/bingham_e${e}_r${r}.pkl"
   out="./scripts/data/compare_newton/bingham_e${e}_r${r}.out"
   cmd="mpiexec -n 16 python $exe $args $bin -logfile $log"
   #echo $cmd
   $cmd | tee $out
done
done

