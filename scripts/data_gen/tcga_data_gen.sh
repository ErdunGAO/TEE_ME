
# declare -a t_noise_dists=("uniform" "normal")
# declare -a me_dists=("normal" "laplace")
# declare -a me_stds=("0.05" "0.15" "0.25" "0.35" "0.1" "0.2" "0.3")
# declare -a seeds=("0" "1" "2" "3" "4" "5" "6" "7" "8" "9")

declare -a t_noise_dists=("normal")
declare -a me_dists=("normal")
declare -a me_stds=("0.15" "0.25" "0.1" "0.2")
declare -a seeds=("0" "1" "2" "3" "4")

total_cores=$(nproc)
free_cores=($(seq 0 $(($total_cores-1))))
python_commands=()

for seed in "${seeds[@]}"
do
    for me_std in "${me_stds[@]}"
    do
        for me_dist in "${me_dists[@]}"
        do
            for t_noise_dist in "${t_noise_dists[@]}"
            do
                echo "seed: $seed, me_std: $me_std, me_dist: $me_dist, t_noise_dist: $t_noise_dist"
                python_command="python data/tcga_generate_data.py --save_dir dataset/tcga/gene_data \
                                                  --me_std $me_std \
                                                  --me_dist $me_dist \
                                                  --t_noise_dist $t_noise_dist \
                                                  --seed $seed"
                python_commands+=("$python_command")
            done
        done
    done
done

# Main loop to iterate through Python commands and assign them to free CPU cores
for command in "${python_commands[@]}"; do
    # Check if there are any free CPU cores
    while [ ${#free_cores[@]} -eq 0 ]; do
        echo "All CPU cores are busy. Waiting for a core to become available..."
        # Sleep for 1 second and recheck
        sleep 1
        for pid in "${!running_pids[@]}"; do
            if ! ps -p $pid > /dev/null; then
                # The Python command has completed, add its core back to the free_cores list
                core=${running_pids[$pid]}
                free_cores+=($core)
                unset running_pids[$pid]
                echo "CPU core $core is now free (PID: $pid)"
            fi
        done
    done

    # Get a free CPU core and remove it from the list
    core=${free_cores[0]}
    unset free_cores[0]
    free_cores=("${free_cores[@]}")

    # Run the Python command on the specified core in the background
    taskset -c $core $command &   
    pid=$!       
    running_pids[$pid]=$core
    echo "Assigned $command to CPU core $core (PID: $pid)"
done

# Wait for all Python commands to complete
wait

echo "All Python commands have completed."