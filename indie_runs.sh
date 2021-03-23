# sh indie_runs.sh
for i in {1..5};
do
    python3 run_goal_pdqn.py --multipass=False;
done