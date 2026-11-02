//! Eco-mode governor: a duty-cycle controller for long batch renders.
//!
//! The worker sleeps `work × (1/duty − 1)` after each GPU step, which caps
//! average GPU power without touching driver clocks (portable to every vendor).

use std::time::Duration;

use fs_types::telemetry::EcoMode;

pub const MIN_DUTY: f32 = 0.1;

/// Duty applied in `Auto` mode on battery when no temperature is available.
const BATTERY_DUTY: f32 = 0.6;

#[derive(Debug, Clone)]
pub struct Governor {
    mode: EcoMode,
    duty: f32,
    integral: f32,
}

impl Governor {
    pub fn new(mode: EcoMode) -> Self {
        Self {
            mode,
            duty: 1.0,
            integral: 0.0,
        }
    }

    pub fn set_mode(&mut self, mode: EcoMode) {
        self.mode = mode;
        self.integral = 0.0;
    }

    pub fn mode(&self) -> EcoMode {
        self.mode
    }

    pub fn duty(&self) -> f32 {
        self.duty
    }

    /// Advance the controller by `dt_s` seconds and return the new duty (0.1–1.0).
    pub fn update(&mut self, temp_c: Option<f32>, on_battery: bool, dt_s: f32) -> f32 {
        self.duty = match self.mode {
            EcoMode::Off => 1.0,
            EcoMode::Fixed { duty_pct } => (f32::from(duty_pct) / 100.0).clamp(MIN_DUTY, 1.0),
            EcoMode::Auto { target_c } => match temp_c {
                Some(t) => {
                    // PI on temperature error. Negative error (too hot) lowers duty.
                    const KP: f32 = 0.08;
                    const KI: f32 = 0.01;
                    let err = target_c - t;
                    // Anti-windup: integral only ever pulls duty down, bounded.
                    self.integral = (self.integral + err * KI * dt_s).clamp(-0.9, 0.0);
                    let cap = if on_battery { BATTERY_DUTY } else { 1.0 };
                    (1.0 + KP * err.min(0.0) + self.integral).clamp(MIN_DUTY, cap)
                }
                None if on_battery => BATTERY_DUTY,
                None => 1.0,
            },
        };
        self.duty
    }

    pub fn is_throttling(&self) -> bool {
        self.duty < 0.999
    }
}

/// Idle time to insert after `work` to achieve `duty`.
pub fn sleep_after(work: Duration, duty: f32) -> Duration {
    let duty = duty.clamp(MIN_DUTY, 1.0);
    if duty >= 0.999 {
        return Duration::ZERO;
    }
    work.mul_f32(1.0 / duty - 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_and_off() {
        let mut g = Governor::new(EcoMode::Fixed { duty_pct: 50 });
        assert_eq!(g.update(None, false, 1.0), 0.5);
        assert_eq!(
            sleep_after(Duration::from_millis(100), 0.5),
            Duration::from_millis(100)
        );
        g.set_mode(EcoMode::Off);
        assert_eq!(g.update(Some(99.0), true, 1.0), 1.0);
        assert_eq!(sleep_after(Duration::from_millis(100), 1.0), Duration::ZERO);
    }

    #[test]
    fn auto_backs_off_when_hot_and_recovers_when_cool() {
        let mut g = Governor::new(EcoMode::Auto { target_c: 80.0 });
        assert_eq!(g.update(Some(70.0), false, 1.0), 1.0, "cool: full speed");
        let mut d = 1.0;
        for _ in 0..60 {
            d = g.update(Some(90.0), false, 1.0);
        }
        assert!(
            d < 0.4,
            "sustained 10 °C over target must throttle hard, got {d}"
        );
        assert!(d >= MIN_DUTY);
        for _ in 0..120 {
            d = g.update(Some(70.0), false, 1.0);
        }
        assert!(d > 0.95, "integral must unwind once cool, got {d}");
    }

    #[test]
    fn battery_caps_auto() {
        let mut g = Governor::new(EcoMode::Auto { target_c: 80.0 });
        assert_eq!(g.update(None, true, 1.0), 0.6);
        assert_eq!(g.update(Some(50.0), true, 1.0), 0.6);
    }
}
