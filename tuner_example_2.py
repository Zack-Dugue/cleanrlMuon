# file: multi_env_tuner_atari.py
import optuna
from cleanrl_utils.tuner import Tuner

# Path to your CleanRL PPO script
CLEANRL_PPO = r"C:\Users\dugue\PycharmProjects\cleanrlMuon\cleanrl\ppo_atari.py"

# Use human-normalized target scores (random, human). Ballpark values are fine.
# 🎯 Tuning set (20 games)
TARGET_SCORES_TUNE = {
    "AlienNoFrameskip-v4": (227.75, 7127.7),
    "AmidarNoFrameskip-v4": (5.77, 1719.5),
    "AssaultNoFrameskip-v4": (222.39, 742.0),
    "AsterixNoFrameskip-v4": (210.0, 8503.3),
    "BoxingNoFrameskip-v4": (0.05, 12.1),
    "BreakoutNoFrameskip-v4": (1.72, 30.5),
    "CentipedeNoFrameskip-v4": (2090.87, 12017.0),
    "DemonAttackNoFrameskip-v4": (152.07, 1971.0),
    "EnduroNoFrameskip-v4": (0.0, 860.5),
    "FreewayNoFrameskip-v4": (0.01, 29.6),
    "FrostbiteNoFrameskip-v4": (65.2, 4334.7),
    "GopherNoFrameskip-v4": (257.6, 2412.5),
    "HeroNoFrameskip-v4": (1026.97, 30826.4),
    "JamesbondNoFrameskip-v4": (29.0, 302.8),
    "KangarooNoFrameskip-v4": (52.0, 3035.0),
    "KrullNoFrameskip-v4": (1598.05, 2665.5),
    "MsPacmanNoFrameskip-v4": (307.3, 6951.6),
    "PongNoFrameskip-v4": (-20.71, 14.6),
    "SeaquestNoFrameskip-v4": (68.4, 42054.7),
    "SpaceInvadersNoFrameskip-v4": (148.3, 1668.7),
}

# 🧪 Evaluation set (37 games)
TARGET_SCORES_EVAL = {
    "AsteroidsNoFrameskip-v4": (719.1, 47388.7),
    "AtlantisNoFrameskip-v4": (12850.0, 29028.1),
    "BankHeistNoFrameskip-v4": (14.2, 753.1),
    "BattleZoneNoFrameskip-v4": (2360.0, 37187.5),
    "BeamRiderNoFrameskip-v4": (363.88, 16926.5),
    "BerzerkNoFrameskip-v4": (123.65, 2630.4),
    "BowlingNoFrameskip-v4": (23.11, 160.7),
    "ChopperCommandNoFrameskip-v4": (811.0, 7387.8),
    "CrazyClimberNoFrameskip-v4": (10780.5, 35829.4),
    "DefenderNoFrameskip-v4": (2874.5, 18688.9),
    "DoubleDunkNoFrameskip-v4": (-18.55, -16.4),
    "FishingDerbyNoFrameskip-v4": (-91.71, -38.7),
    "GravitarNoFrameskip-v4": (173.0, 3351.4),
    "IceHockeyNoFrameskip-v4": (-11.15, 0.9),
    "KungFuMasterNoFrameskip-v4": (258.5, 22736.3),
    "MontezumaRevengeNoFrameskip-v4": (0.0, 4753.3),
    "NameThisGameNoFrameskip-v4": (2292.35, 8049.0),
    "PhoenixNoFrameskip-v4": (761.4, 7242.6),
    "PitfallNoFrameskip-v4": (-229.44, 6463.7),
    "PrivateEyeNoFrameskip-v4": (24.94, 69571.3),
    "QbertNoFrameskip-v4": (163.88, 13455.0),
    "RiverraidNoFrameskip-v4": (1338.5, 17118.0),
    "RoadRunnerNoFrameskip-v4": (11.5, 7845.0),
    "RobotankNoFrameskip-v4": (2.16, 11.9),
    "SkiingNoFrameskip-v4": (-17098.09, -4336.9),
    "SolarisNoFrameskip-v4": (1236.3, 12326.7),
    "StarGunnerNoFrameskip-v4": (664.0, 10250.0),
    "SurroundNoFrameskip-v4": (-9.99, 6.53),
    "TennisNoFrameskip-v4": (-23.84, -8.3),
    "TimePilotNoFrameskip-v4": (3568.0, 5229.2),
    "TutankhamNoFrameskip-v4": (11.43, 167.6),
    "UpNDownNoFrameskip-v4": (533.4, 11693.2),
    "VentureNoFrameskip-v4": (0.0, 1187.5),
    "VideoPinballNoFrameskip-v4": (0.0, 17667.9),
    "WizardOfWorNoFrameskip-v4": (563.5, 4756.5),
    "YarsRevengeNoFrameskip-v4": (3092.91, 54576.9),
    "ZaxxonNoFrameskip-v4": (32.5, 9173.3),
}


tuner = Tuner(
    script=CLEANRL_PPO,
    metric="charts/episodic_return",
    metric_last_n_average_window=50,
    direction="maximize",
    aggregation_type="median",        # aggregate across envs
    target_scores=TARGET_SCORES_TUNE,      # enables human-normalized scoring
    pruner=optuna.pruners.MedianPruner(n_startup_trials=3),
    sampler=optuna.samplers.TPESampler(seed=0),
    params_fn=lambda trial: {
        # ---- MUST-TUNE set ----
        "learning-rate": trial.suggest_float("lr", 3e-5, 3e-3, log=True),
        "ent-coef": trial.suggest_float("ent_coef", 0.0, 0.02),
        "update-epochs": trial.suggest_int("update_epochs", 2, 8),

        # ---- FIXED params go here (no fixed_params kw) ----
        # Pick one total batch size; you can also expose this as a categorical if desired.
        "num-envs": 32,
        "num-steps": 256,              # total batch = 8192
        "total-timesteps": 5_000_000,
        # Optional: uncomment to save disk
        # "capture-video": False,
        # If you patched PPO to accept optimizer choices/Betas, include them here too.
    },
)

# Launch the study: N trials, averaging across num_seeds for each env
tuner.tune(
    num_trials=10,      # increase for real sweeps
    num_seeds=3,
)
