extern crate autodiff as ad;
extern crate num_traits;
extern crate rand;

use ad::{diff, grad, F};
use num_traits::Float;

#[test]
fn num_test() {
    // Trivial tests without derivatives, which check the F api.
    let x = F::var(1.0);
    assert_eq!(x.x, 1.0);
    assert_eq!(x.dx, 1.0);
    assert_eq!(x.x, x.value());
    assert_eq!(x.dx, x.deriv());

    let y = F::cst(2.0);
    assert_eq!(y.x, y.value());
    assert_eq!(y.dx, y.deriv());
}

// NOTE: we don't need approximate equality here because we compare with the chain rule derivative
// expression, which means that the derivative and expected values should be identically computed.
#[test]
fn simple_test() {
    assert_eq!(diff(|_| F::cst(1.0), 0.0), 0.0);
    assert_eq!(diff(|x| x, 0.0), 1.0);
    assert_eq!(diff(|x| x * x, 0.0), 0.0);
    assert_eq!(diff(|x| x * x, 1.0), 2.0);
    assert_eq!(diff(|x| Float::exp(-x * x / 2.0), 0.0), 0.0);
}

#[test]
fn random_test() {
    use self::rand::{distributions::Uniform, Rng, SeedableRng, StdRng};

    let seed = [3; 32];
    let mut rng: StdRng = SeedableRng::from_seed(seed);
    let range = Uniform::new(-1.0, 1.0);
    for _ in 0..99 {
        let t = rng.sample(range);
        assert_eq!(
            diff(|x| Float::exp(-x * x / F::cst(2.0)), t),
            -t * Float::exp(-t * t / 2.0)
        );
    }
}

#[test]
fn grad_test() {
    use self::rand::{distributions::Uniform, Rng, SeedableRng, StdRng};

    let seed = [3; 32];
    let mut rng: StdRng = SeedableRng::from_seed(seed);
    let range = Uniform::new(-1.0, 1.0);
    for _ in 0..99 {
        let t = vec![rng.sample(range), rng.sample(range)];
        let expected = vec![
            -0.5 * t[1] * Float::exp(-t[0] * t[1] / 2.0),
            -0.5 * t[0] * Float::exp(-t[0] * t[1] / 2.0),
        ];
        assert_eq!(grad(|x| Float::exp(-x[0] * x[1] / 2.0), &t), expected);
    }
}
