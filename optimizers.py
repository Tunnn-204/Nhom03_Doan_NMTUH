import itertools
import numpy as np
import optuna
from config import SEARCH_SPACE, DEFAULT_PARAMS

optuna.logging.set_verbosity(optuna.logging.WARNING)


def sample_random_point(rng: np.random.Generator) -> dict:
    """Lấy mẫu ngẫu nhiên một vector siêu tham số x trong miền SEARCH_SPACE."""
    return {
        "n_estimators": int(rng.integers(*SEARCH_SPACE["n_estimators"], endpoint=True)),
        "max_depth": int(rng.integers(*SEARCH_SPACE["max_depth"], endpoint=True)),
        "learning_rate": float(
            np.exp(rng.uniform(np.log(SEARCH_SPACE["learning_rate"][0]), np.log(SEARCH_SPACE["learning_rate"][1])))
        ),
        "subsample": float(rng.uniform(*SEARCH_SPACE["subsample"])),
        "colsample_bytree": float(rng.uniform(*SEARCH_SPACE["colsample_bytree"])),
        "reg_lambda": float(
            np.exp(rng.uniform(np.log(SEARCH_SPACE["reg_lambda"][0]), np.log(SEARCH_SPACE["reg_lambda"][1])))
        ),
        "scale_pos_weight": float(rng.uniform(*SEARCH_SPACE["scale_pos_weight"])),
    }


def clip_point(params: dict) -> dict:
    """Ép nghiệm nằm trong biên hợp lệ của không gian tìm kiếm Omega."""
    clipped = {}
    for k, (low, high) in SEARCH_SPACE.items():
        val = np.clip(params[k], low, high)
        clipped[k] = int(round(val)) if k in ["n_estimators", "max_depth"] else float(val)
    return clipped


# 1. Baseline 0: Cấu hình mặc định của thư viện XGBoost
def run_default_baseline(obj_func):
    obj_func.evaluate(DEFAULT_PARAMS)
    return obj_func.best_params, obj_func.trials_log


# 2. Baseline 1: Grid Search (Tìm kiếm lưới điểm cố định trong ngân sách n_trials)
def run_grid_search(obj_func, n_trials: int):
    grid_axes = {
        "n_estimators": [80, 220],
        "max_depth": [4, 8],
        "learning_rate": [0.03, 0.15],
        "subsample": [0.7, 0.9],
        "colsample_bytree": [0.7, 0.9],
        "reg_lambda": [0.1, 2.0],
        "scale_pos_weight": [1.2, 3.0],
    }
    keys = list(grid_axes.keys())
    all_combos = list(itertools.product(*(grid_axes[k] for k in keys)))
    indices = np.linspace(0, len(all_combos) - 1, num=min(n_trials, len(all_combos)), dtype=int)

    for idx in indices:
        params = dict(zip(keys, all_combos[idx]))
        obj_func.evaluate(params)

    return obj_func.best_params, obj_func.trials_log


# 3. Baseline 2: Random Search
def run_random_search(obj_func, n_trials: int, seed: int):
    rng = np.random.default_rng(seed)
    for _ in range(n_trials):
        params = sample_random_point(rng)
        obj_func.evaluate(params)
    return obj_func.best_params, obj_func.trials_log


# 4. Phương pháp chính 1: Bayesian Optimization với TPE (Tree-structured Parzen Estimator)
def run_tpe_optuna(obj_func, n_trials: int, seed: int):
    sampler = optuna.samplers.TPESampler(seed=seed, n_startup_trials=max(5, n_trials // 4))
    study = optuna.create_study(direction="maximize", sampler=sampler)

    def _optuna_objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", *SEARCH_SPACE["n_estimators"]),
            "max_depth": trial.suggest_int("max_depth", *SEARCH_SPACE["max_depth"]),
            "learning_rate": trial.suggest_float("learning_rate", *SEARCH_SPACE["learning_rate"], log=True),
            "subsample": trial.suggest_float("subsample", *SEARCH_SPACE["subsample"]),
            "colsample_bytree": trial.suggest_float("colsample_bytree", *SEARCH_SPACE["colsample_bytree"]),
            "reg_lambda": trial.suggest_float("reg_lambda", *SEARCH_SPACE["reg_lambda"], log=True),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", *SEARCH_SPACE["scale_pos_weight"]),
        }
        return obj_func.evaluate(params)

    study.optimize(_optuna_objective, n_trials=n_trials, show_progress_bar=False)
    return obj_func.best_params, obj_func.trials_log


# 5. Phương pháp chính 2: Genetic Algorithm (Thuật toán Di truyền tự cài đặt)
def run_genetic_algorithm(
    obj_func,
    n_trials: int,
    seed: int,
    pop_size: int = 8,
    crossover_rate: float = 0.8,
    mutation_rate: float = 0.25
):
    rng = np.random.default_rng(seed)
    pop_size = min(pop_size, n_trials)

    # Khởi tạo quần thể thế hệ đầu tiên
    population = [sample_random_point(rng) for _ in range(pop_size)]
    fitness = [obj_func.evaluate(ind) for ind in population]
    evals_used = pop_size

    while evals_used < n_trials:
        new_population = []
        # Elitism: Giữ lại cá thể tốt nhất từ thế hệ trước
        best_idx = int(np.argmax(fitness))
        new_population.append(population[best_idx].copy())

        while len(new_population) < pop_size and (evals_used + len(new_population) - 1) < n_trials:
            # Chọn lọc giải đấu (Tournament Selection bậc 2)
            i1, i2 = rng.integers(0, len(population), size=2)
            parent1 = population[i1] if fitness[i1] >= fitness[i2] else population[i2]

            j1, j2 = rng.integers(0, len(population), size=2)
            parent2 = population[j1] if fitness[j1] >= fitness[j2] else population[j2]

            # Lai ghép (Arithmetic Crossover)
            child = {}
            for k in SEARCH_SPACE.keys():
                if rng.random() < crossover_rate:
                    alpha = rng.uniform(0.2, 0.8)
                    child[k] = alpha * parent1[k] + (1.0 - alpha) * parent2[k]
                else:
                    child[k] = parent1[k]

            # Đột biến Gaussian (Gaussian Mutation)
            for k, (low, high) in SEARCH_SPACE.items():
                if rng.random() < mutation_rate:
                    sigma = (high - low) * 0.15
                    child[k] += rng.normal(0.0, sigma)

            new_population.append(clip_point(child))

        # Đánh giá độ thích nghi của các cá thể mới
        new_fitness = [fitness[best_idx]]
        for ind in new_population[1:]:
            if evals_used >= n_trials:
                break
            new_fitness.append(obj_func.evaluate(ind))
            evals_used += 1

        population = new_population[:len(new_fitness)]
        fitness = new_fitness

    return obj_func.best_params, obj_func.trials_log