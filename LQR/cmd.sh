setting=1

for METHOD in "PDFL2" "Energy" "RBF" "Laplace" "KL" "FLE"
do
    for N in {300..1000..50}
    do
        nohup python -u run.py --method "$METHOD" --N_size "$N" --setting "$setting" --simulation_num 50 >> process_${METHOD}_setting${setting}.txt
    done
done
python comparison_visual.py --setting "$setting"
python comparison_table.py --setting "$setting"

