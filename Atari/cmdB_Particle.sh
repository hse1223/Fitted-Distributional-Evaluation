# Atlantis

GAME="AtlantisNoFrameskip-v4"

nohup python -u B1_truedistribution.py --game "$GAME" --return_samples 1000  --epsilon_target 0.0  --reward_var 1.0 >> "${GAME}_generating.txt"
nohup python -u B1_truedistribution.py --game "$GAME" --return_samples 1000  --epsilon_target 0.3  --reward_var 0.0 >> "${GAME}_generating.txt"

for SAMPLESIZE in 2000 # 5000 # 10000
do
    for SEED in 1 2 3 4 5
    do
        for METHOD in QRDQN IQN
        do
            for NUMMIX in 10 100 200
            do
                nohup python -u B3_Particle.py --game "$GAME" --num_mixture "$NUMMIX"  --epsilon_target 0.0  --reward_var 1.0 --epsilon_behavior 0.1 --seedno "$SEED" --sample_size "$SAMPLESIZE" --Num_Iterations 50000 --method "$METHOD" >> "${GAME}_${METHOD}_N${SAMPLESIZE}_epsA.txt"
                nohup python -u B3_Particle.py --game "$GAME" --num_mixture "$NUMMIX"  --epsilon_target 0.0  --reward_var 1.0 --epsilon_behavior 0.5 --seedno "$SEED" --sample_size "$SAMPLESIZE" --Num_Iterations 50000 --method "$METHOD" >> "${GAME}_${METHOD}_N${SAMPLESIZE}_epsB.txt"
                nohup python -u B3_Particle.py --game "$GAME" --num_mixture "$NUMMIX"  --epsilon_target 0.3  --reward_var 0.0 --epsilon_behavior 0.4 --seedno "$SEED" --sample_size "$SAMPLESIZE" --Num_Iterations 50000 --method "$METHOD" >> "${GAME}_${METHOD}_N${SAMPLESIZE}_epsC.txt"
                nohup python -u B3_Particle.py --game "$GAME" --num_mixture "$NUMMIX"  --epsilon_target 0.3  --reward_var 0.0 --epsilon_behavior 0.8 --seedno "$SEED" --sample_size "$SAMPLESIZE" --Num_Iterations 50000 --method "$METHOD" >> "${GAME}_${METHOD}_N${SAMPLESIZE}_epsD.txt"
            done
        done
    done
done



# Breakout (target = 0.3)

GAME="BreakoutNoFrameskip-v4"

nohup python -u B1_truedistribution.py --game "$GAME" --return_samples 1000  --epsilon_target 0.3  --reward_var 1.0 >> "${GAME}_generating.txt"
nohup python -u B1_truedistribution.py --game "$GAME" --return_samples 1000  --epsilon_target 0.3  --reward_var 0.0 >> "${GAME}_generating.txt"

for SAMPLESIZE in 2000 # 5000 # 10000
do
    for SEED in 1 2 3 4 5
    do
        for METHOD in QRDQN IQN
        do
            for NUMMIX in 10 100 200
            do
                nohup python -u B3_Particle.py --game "$GAME" --num_mixture "$NUMMIX"  --epsilon_target 0.3  --reward_var 1.0 --epsilon_behavior 0.1 --seedno "$SEED" --sample_size "$SAMPLESIZE" --Num_Iterations 50000 --method "$METHOD" >> "${GAME}_${METHOD}_N${SAMPLESIZE}_epsA.txt"
                nohup python -u B3_Particle.py --game "$GAME" --num_mixture "$NUMMIX"  --epsilon_target 0.3  --reward_var 1.0 --epsilon_behavior 0.5 --seedno "$SEED" --sample_size "$SAMPLESIZE" --Num_Iterations 50000 --method "$METHOD" >> "${GAME}_${METHOD}_N${SAMPLESIZE}_epsB.txt"
                nohup python -u B3_Particle.py --game "$GAME" --num_mixture "$NUMMIX"  --epsilon_target 0.3  --reward_var 0.0 --epsilon_behavior 0.4 --seedno "$SEED" --sample_size "$SAMPLESIZE" --Num_Iterations 50000 --method "$METHOD" >> "${GAME}_${METHOD}_N${SAMPLESIZE}_epsC.txt"
                nohup python -u B3_Particle.py --game "$GAME" --num_mixture "$NUMMIX"  --epsilon_target 0.3  --reward_var 0.0 --epsilon_behavior 0.8 --seedno "$SEED" --sample_size "$SAMPLESIZE" --Num_Iterations 50000 --method "$METHOD" >> "${GAME}_${METHOD}_N${SAMPLESIZE}_epsD.txt"
            done
        done
    done
done


# Enduro (behavior <= 0.5)

GAME="EnduroNoFrameskip-v4"

nohup python -u B1_truedistribution.py --game "$GAME" --return_samples 1000  --epsilon_target 0.0  --reward_var 1.0 >> "${GAME}_generating.txt"
nohup python -u B1_truedistribution.py --game "$GAME" --return_samples 1000  --epsilon_target 0.3  --reward_var 0.0 >> "${GAME}_generating.txt"

for SAMPLESIZE in 2000 # 5000 # 10000
do
    for SEED in 1 2 3 4 5
    do
        for METHOD in QRDQN IQN
        do
            for NUMMIX in 10 100 200
            do
                nohup python -u B3_Particle.py --game "$GAME" --num_mixture "$NUMMIX"  --epsilon_target 0.0  --reward_var 1.0 --epsilon_behavior 0.1 --seedno "$SEED" --sample_size "$SAMPLESIZE" --Num_Iterations 50000 --method "$METHOD" >> "${GAME}_${METHOD}_N${SAMPLESIZE}_epsA.txt"
                nohup python -u B3_Particle.py --game "$GAME" --num_mixture "$NUMMIX"  --epsilon_target 0.0  --reward_var 1.0 --epsilon_behavior 0.5 --seedno "$SEED" --sample_size "$SAMPLESIZE" --Num_Iterations 50000 --method "$METHOD" >> "${GAME}_${METHOD}_N${SAMPLESIZE}_epsB.txt"
                nohup python -u B3_Particle.py --game "$GAME" --num_mixture "$NUMMIX"  --epsilon_target 0.3  --reward_var 0.0 --epsilon_behavior 0.4 --seedno "$SEED" --sample_size "$SAMPLESIZE" --Num_Iterations 50000 --method "$METHOD" >> "${GAME}_${METHOD}_N${SAMPLESIZE}_epsC.txt"
                nohup python -u B3_Particle.py --game "$GAME" --num_mixture "$NUMMIX"  --epsilon_target 0.3  --reward_var 0.0 --epsilon_behavior 0.5 --seedno "$SEED" --sample_size "$SAMPLESIZE" --Num_Iterations 50000 --method "$METHOD" >> "${GAME}_${METHOD}_N${SAMPLESIZE}_epsD.txt"
            done
        done
    done
done



# Kungfu

GAME="KungFuMasterNoFrameskip-v4"

nohup python -u B1_truedistribution.py --game "$GAME" --return_samples 1000  --epsilon_target 0.0  --reward_var 1.0 >> "${GAME}_generating.txt"
nohup python -u B1_truedistribution.py --game "$GAME" --return_samples 1000  --epsilon_target 0.3  --reward_var 0.0 >> "${GAME}_generating.txt"

for SAMPLESIZE in 2000 # 5000 # 10000
do
    for SEED in 1 2 3 4 5
    do
        for METHOD in QRDQN IQN
        do
            for NUMMIX in 10 100 200
            do
                nohup python -u B3_Particle.py --game "$GAME" --num_mixture "$NUMMIX"  --epsilon_target 0.0  --reward_var 1.0 --epsilon_behavior 0.1 --seedno "$SEED" --sample_size "$SAMPLESIZE" --Num_Iterations 50000 --method "$METHOD" >> "${GAME}_${METHOD}_N${SAMPLESIZE}_epsA.txt"
                nohup python -u B3_Particle.py --game "$GAME" --num_mixture "$NUMMIX"  --epsilon_target 0.0  --reward_var 1.0 --epsilon_behavior 0.5 --seedno "$SEED" --sample_size "$SAMPLESIZE" --Num_Iterations 50000 --method "$METHOD" >> "${GAME}_${METHOD}_N${SAMPLESIZE}_epsB.txt"
                nohup python -u B3_Particle.py --game "$GAME" --num_mixture "$NUMMIX"  --epsilon_target 0.3  --reward_var 0.0 --epsilon_behavior 0.4 --seedno "$SEED" --sample_size "$SAMPLESIZE" --Num_Iterations 50000 --method "$METHOD" >> "${GAME}_${METHOD}_N${SAMPLESIZE}_epsC.txt"
                nohup python -u B3_Particle.py --game "$GAME" --num_mixture "$NUMMIX"  --epsilon_target 0.3  --reward_var 0.0 --epsilon_behavior 0.8 --seedno "$SEED" --sample_size "$SAMPLESIZE" --Num_Iterations 50000 --method "$METHOD" >> "${GAME}_${METHOD}_N${SAMPLESIZE}_epsD.txt"
            done
        done
    done
done



# Pong (behavior <= 0.5, random_var1.0, eps-bhavior0.3)

GAME="PongNoFrameskip-v4"

nohup python -u B1_truedistribution.py --game "$GAME" --return_samples 1000  --epsilon_target 0.0  --reward_var 1.0 >> "${GAME}_generating.txt"
nohup python -u B1_truedistribution.py --game "$GAME" --return_samples 1000  --epsilon_target 0.3  --reward_var 0.0 >> "${GAME}_generating.txt"

for SAMPLESIZE in 2000 # 5000 # 10000
do
    for SEED in 1 2 3 4 5
    do
        for METHOD in QRDQN IQN
        do
            for NUMMIX in 10 100 200
            do
                nohup python -u B3_Particle.py --game "$GAME" --num_mixture "$NUMMIX"  --epsilon_target 0.0  --reward_var 1.0 --epsilon_behavior 0.1 --seedno "$SEED" --sample_size "$SAMPLESIZE" --Num_Iterations 50000 --method "$METHOD" >> "${GAME}_${METHOD}_N${SAMPLESIZE}_epsA.txt"
                nohup python -u B3_Particle.py --game "$GAME" --num_mixture "$NUMMIX"  --epsilon_target 0.0  --reward_var 1.0 --epsilon_behavior 0.3 --seedno "$SEED" --sample_size "$SAMPLESIZE" --Num_Iterations 50000 --method "$METHOD" >> "${GAME}_${METHOD}_N${SAMPLESIZE}_epsB.txt"
                nohup python -u B3_Particle.py --game "$GAME" --num_mixture "$NUMMIX"  --epsilon_target 0.3  --reward_var 0.0 --epsilon_behavior 0.4 --seedno "$SEED" --sample_size "$SAMPLESIZE" --Num_Iterations 50000 --method "$METHOD" >> "${GAME}_${METHOD}_N${SAMPLESIZE}_epsC.txt"
                nohup python -u B3_Particle.py --game "$GAME" --num_mixture "$NUMMIX"  --epsilon_target 0.3  --reward_var 0.0 --epsilon_behavior 0.5 --seedno "$SEED" --sample_size "$SAMPLESIZE" --Num_Iterations 50000 --method "$METHOD" >> "${GAME}_${METHOD}_N${SAMPLESIZE}_epsD.txt"
            done
        done
    done
done



# Qbert

GAME="QbertNoFrameskip-v4"

nohup python -u B1_truedistribution.py --game "$GAME" --return_samples 1000  --epsilon_target 0.0  --reward_var 1.0 >> "${GAME}_generating.txt"
nohup python -u B1_truedistribution.py --game "$GAME" --return_samples 1000  --epsilon_target 0.3  --reward_var 0.0 >> "${GAME}_generating.txt"


for SAMPLESIZE in 2000 # 5000 # 10000
do
    for SEED in 1 2 3 4 5
    do
        for METHOD in QRDQN IQN
        do
            for NUMMIX in 10 100 200
            do
                nohup python -u B3_Particle.py --game "$GAME" --num_mixture "$NUMMIX"  --epsilon_target 0.0  --reward_var 1.0 --epsilon_behavior 0.1 --seedno "$SEED" --sample_size "$SAMPLESIZE" --Num_Iterations 50000 --method "$METHOD" >> "${GAME}_${METHOD}_N${SAMPLESIZE}_epsA.txt"
                nohup python -u B3_Particle.py --game "$GAME" --num_mixture "$NUMMIX"  --epsilon_target 0.0  --reward_var 1.0 --epsilon_behavior 0.5 --seedno "$SEED" --sample_size "$SAMPLESIZE" --Num_Iterations 50000 --method "$METHOD" >> "${GAME}_${METHOD}_N${SAMPLESIZE}_epsB.txt"
                nohup python -u B3_Particle.py --game "$GAME" --num_mixture "$NUMMIX"  --epsilon_target 0.3  --reward_var 0.0 --epsilon_behavior 0.4 --seedno "$SEED" --sample_size "$SAMPLESIZE" --Num_Iterations 50000 --method "$METHOD" >> "${GAME}_${METHOD}_N${SAMPLESIZE}_epsC.txt"
                nohup python -u B3_Particle.py --game "$GAME" --num_mixture "$NUMMIX"  --epsilon_target 0.3  --reward_var 0.0 --epsilon_behavior 0.8 --seedno "$SEED" --sample_size "$SAMPLESIZE" --Num_Iterations 50000 --method "$METHOD" >> "${GAME}_${METHOD}_N${SAMPLESIZE}_epsD.txt"
            done
        done
    done
done




# Spaceinvader

GAME="SpaceInvadersNoFrameskip-v4"

nohup python -u B1_truedistribution.py --game "$GAME" --return_samples 1000  --epsilon_target 0.0  --reward_var 1.0 >> "${GAME}_generating.txt"
nohup python -u B1_truedistribution.py --game "$GAME" --return_samples 1000  --epsilon_target 0.3  --reward_var 0.0 >> "${GAME}_generating.txt"

for SAMPLESIZE in 2000 # 5000 # 10000
do
    for SEED in 1 2 3 4 5
    do
        for METHOD in QRDQN IQN
        do
            for NUMMIX in 10 100 200
            do
                nohup python -u B3_Particle.py --game "$GAME" --num_mixture "$NUMMIX"  --epsilon_target 0.0  --reward_var 1.0 --epsilon_behavior 0.1 --seedno "$SEED" --sample_size "$SAMPLESIZE" --Num_Iterations 50000 --method "$METHOD" >> "${GAME}_${METHOD}_N${SAMPLESIZE}_epsA.txt"
                nohup python -u B3_Particle.py --game "$GAME" --num_mixture "$NUMMIX"  --epsilon_target 0.0  --reward_var 1.0 --epsilon_behavior 0.5 --seedno "$SEED" --sample_size "$SAMPLESIZE" --Num_Iterations 50000 --method "$METHOD" >> "${GAME}_${METHOD}_N${SAMPLESIZE}_epsB.txt"
                nohup python -u B3_Particle.py --game "$GAME" --num_mixture "$NUMMIX"  --epsilon_target 0.3  --reward_var 0.0 --epsilon_behavior 0.4 --seedno "$SEED" --sample_size "$SAMPLESIZE" --Num_Iterations 50000 --method "$METHOD" >> "${GAME}_${METHOD}_N${SAMPLESIZE}_epsC.txt"
                nohup python -u B3_Particle.py --game "$GAME" --num_mixture "$NUMMIX"  --epsilon_target 0.3  --reward_var 0.0 --epsilon_behavior 0.8 --seedno "$SEED" --sample_size "$SAMPLESIZE" --Num_Iterations 50000 --method "$METHOD" >> "${GAME}_${METHOD}_N${SAMPLESIZE}_epsD.txt"
            done
        done
    done
done




