# Train the following games. 

nohup python -u A1_dqn_train.py --game AtlantisNoFrameskip-v4 --setting nature2 >> game1.txt
nohup python -u  A1_dqn_train.py --game BreakoutNoFrameskip-v4 --setting nature2 >> game2.txt
nohup python -u  A1_dqn_train.py --game EnduroNoFrameskip-v4 --setting nature2 >> game3.txt
nohup python -u  A1_dqn_train.py --game KungFuMasterNoFrameskip-v4 --setting nature2 >> game4.txt
nohup python -u  A1_dqn_train.py --game PongNoFrameskip-v4 --setting nature2 >> game5.txt
nohup python -u  A1_dqn_train.py --game QbertNoFrameskip-v4 --setting nature2 >> game6.txt
nohup python -u  A1_dqn_train.py --game SpaceInvadersNoFrameskip-v4 --setting nature2 >> game7.txt


# Watch the epsilon policy playing (how often it yields rewards).

python A2_dqn_observe.py --game AtlantisNoFrameskip-v4 --epsilon 0.8
python A2_dqn_observe.py --game BreakoutNoFrameskip-v4 --epsilon 0.8
python A2_dqn_observe.py --game EnduroNoFrameskip-v4 --epsilon 0.5
python A2_dqn_observe.py --game KungFuMasterNoFrameskip-v4 --epsilon 0.8
python A2_dqn_observe.py --game PongNoFrameskip-v4 --epsilon 0.5
python A2_dqn_observe.py --game QbertNoFrameskip-v4 --epsilon 0.8
python A2_dqn_observe.py --game SpaceInvadersNoFrameskip-v4 --epsilon 0.8




