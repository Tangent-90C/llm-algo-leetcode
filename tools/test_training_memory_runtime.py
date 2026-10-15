from __future__ import annotations

from tools.training_memory_runtime import measure_phased_training_run


class FakeCuda:
    def __init__(self) -> None:
        self.allocated = 0
        self.reserved = 0
        self.peak_allocated = 0
        self.peak_reserved = 0

    def is_available(self) -> bool:
        return True

    def synchronize(self) -> None:
        return None

    def reset_peak_memory_stats(self) -> None:
        self.peak_allocated = self.allocated
        self.peak_reserved = self.reserved

    def memory_allocated(self) -> int:
        return self.allocated

    def memory_reserved(self) -> int:
        return self.reserved

    def max_memory_allocated(self) -> int:
        return self.peak_allocated

    def max_memory_reserved(self) -> int:
        return self.peak_reserved

    def set_memory(self, allocated_mb: int, reserved_mb: int) -> None:
        self.allocated = allocated_mb * 1024**2
        self.reserved = reserved_mb * 1024**2
        self.peak_allocated = max(self.peak_allocated, self.allocated)
        self.peak_reserved = max(self.peak_reserved, self.reserved)


class FakeTorch:
    def __init__(self) -> None:
        self.cuda = FakeCuda()


def main() -> None:
    torch_module = FakeTorch()

    def train_step(record_phase):
        torch_module.cuda.set_memory(300, 350)
        record_phase("forward", lambda: torch_module.cuda.set_memory(400, 450))
        record_phase("backward", lambda: torch_module.cuda.set_memory(500, 560))
        record_phase("optimizer", lambda: torch_module.cuda.set_memory(520, 600))
        return 1.25

    result = measure_phased_training_run(
        train_step,
        torch_module=torch_module,
        batch_size=1,
        warmup=1,
        iters=2,
    )
    assert result["global_peak_phase"] == "optimizer"
    assert result["phase_metrics"]["forward"]["peak_allocated_mb"] == 400.0
    assert result["phase_metrics"]["backward"]["peak_allocated_mb"] == 500.0
    assert result["phase_metrics"]["optimizer"]["peak_reserved_mb"] == 600.0
    assert result["measurement_role"] == "phase_memory_attribution_only"
    assert result["timing_for_adoption"] is False
    print("training memory runtime: ok")


if __name__ == "__main__":
    main()
