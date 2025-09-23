import numpy as np

class TumorGrowthModel:
    """
    Simple Lottka-Volterra tumor growth simulator
    with ramped growth/death and optional noise.
    """

    def __init__(
        self,
        initial_wt: float = 0.99,
        initial_mut: float = 0.01,
        growth_rate_wt: float = 0.175,
        growth_rate_mut: float = 0.175,
        death_rate_wt: float = 0.001,
        death_rate_mut: float = 0.001,
        treat_death_rate_wt: float = 0.15,
        treat_death_rate_mut: float = 0.0,
        carrying_capacity: float = 100000,
        competition_wt: float = 2.0,
        competition_mut: float = 1.0,
        timestep_size: float = 0.005,
        ramp_time: int = 4.5,
        max_time: int = 300000,
        noise: bool = True,
        treatment_timestep: int = 800,
    ):
        self.state = np.array([initial_wt, initial_mut, 0.0])  # [WT, MUT, treatment_flag]
        self.growth_rate = np.array([growth_rate_wt, growth_rate_mut])
        self.death_rate = np.array([death_rate_wt, death_rate_mut])
        self.treat_death_rate = np.array([treat_death_rate_wt, treat_death_rate_mut])
        self.capacity = carrying_capacity
        self.competition = np.array([competition_wt, competition_mut])
        self.timestep_size = timestep_size
        self.ramp_time = ramp_time
        self.max_time = max_time
        self.noise = noise
        self.treatment_timestep = treatment_timestep

        # dynamic fractions for ramped effect
        self.growth_fraction = 1.0
        self.death_fraction = 0.0

        self.time = 0

    def _update_fractions(self, action: int):
        """Update ramped growth/death fractions depending on treatment."""
        step = self.timestep_size / self.ramp_time

        if action:  # treatment ON
            if self.growth_fraction > 0:
                self.growth_fraction = max(0, self.growth_fraction - step)
            else:
                self.death_fraction = min(1, self.death_fraction + step)
        else:  # treatment OFF
            if self.death_fraction > 0:
                self.death_fraction = max(0, self.death_fraction - step)
            else:
                self.growth_fraction = min(1, self.growth_fraction + step)

    def _grow(self, i: int, j: int, action: int) -> float:
        """Simulate one growth step for population i."""
        # update growth/death rate fractions
        current_growth_rate = np.array([
            self.growth_rate[0] * self.growth_fraction,
            self.growth_rate[1]
        ])
        current_death_rate_treat = np.array([
            self.treat_death_rate[0] * self.death_fraction,
            self.treat_death_rate[1]
        ])

        # L-V logistic growth with treatment
        new_pop = self.state[i] * (
            1
            + self.timestep_size * current_growth_rate[i] *
            (1 - (self.state[i] + self.state[j] * self.competition[j]) / self.capacity)
            * (1 - current_death_rate_treat[i] * action)
            - current_growth_rate[i] * self.death_rate[i] * self.timestep_size
        )

        # add noise
        if self.noise and new_pop > 0:
            rand = np.random.normal(0, 0.01 * new_pop)
            rand = np.clip(rand, -0.05 * new_pop, 0.05 * new_pop)
            new_pop += rand

        return max(new_pop, 0)

    def step(self, action: int):
        """Perform one timestep."""

        self.state[2] = action

        for t in range(self.treatment_timestep):
            # update fractions (ramped effect)
            self.time += 1
            self._update_fractions(action)

            # update populations
            wt_new = self._grow(0, 1, action)
            mut_new = self._grow(1, 0, action)

            self.state[0], self.state[1] = wt_new, mut_new
            #self.trajectory[:, self.time] = self.state

    def run(self, treatment_schedule: np.ndarray):
        """
        Run full trajectory given treatment schedule (0/1 array).
        """
        for t in range(self.max_time):
            self.step(treatment_schedule[t])
        return


if __name__ == "__main__":
    np.random.seed(42)

    # Example: constant treatment ON after day 100
    max_time = 50000
    schedule = np.zeros(max_time + 1)
    schedule[100:] = 1

    model = TumorGrowthModel(max_time=max_time, noise=True)
    for i in range(7):
        model.step(0)
        print(f"Step {i+1}, State: {model.state}")

