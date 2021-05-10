use autodiff::{diff, grad, F, F1};
use num_traits::Float;
use rand::distributions::Uniform;
use rand::prelude::*;

#[test]
fn num_test() {
    // Trivial tests without derivatives, which check the F api.
    let x = F1::var(1.0);
    assert_eq!(x.x, 1.0);
    assert_eq!(x.dx, 1.0);
    assert_eq!(x.x, x.value());
    assert_eq!(x.dx, x.deriv());

    let y = F1::cst(2.0);
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

//#[test]
//fn cubic() {
//    let x: f64 = 0.0;
//    let f1 = |x: F3| (x - 1.0f64) * (x - 1.0f64) * (x - 1.0f64);
//    let f2 = |x: F3| (x - 1.0f64).powi(3);
//
//    let df1 = f1(F3::var(x));
//    let df2 = f2(F3::var(x));
//
//    println!("f(x) = (x-1)^3");
//    assert_eq!(df1.deriv().value(), 3.0);
//    assert_eq!(df1.deriv().deriv().value(), -6.0);
//    assert_eq!(df1.deriv().deriv().deriv(), 6.0);
//    assert_eq!(df2.deriv().value(), 3.0);
//    assert_eq!(df2.deriv().deriv().value(), -6.0);
//    assert_eq!(df2.deriv().deriv().deriv(), 6.0);
//}

//#[test]
//fn third_order_stress_test() {
//    use approx::*;
//    let x: f64 = 0.1;
//    let f = |x: F3| -> F3 {
//        (x * 2.0f64 - (x / 0.5f64).mul_add(x % 3.0f64, x * 1.1f64) + 1.2_f64)
//            .sin()
//            .cos()
//            .abs()
//            .ln()
//            .abs()
//            .sqrt()
//            .pow(x + 6.0_f64)
//            .log(x + 2.1f64)
//            .abs()
//            .log2()
//            .log10()
//            //.abs_sub(x * x + 0.1_f64)
//            .tan()
//            .asin()
//            .acos()
//            .atan()
//            .atan2(x + 1.0f64)
//            .sinh()
//            .cosh()
//            .tanh()
//            .pow(2f64)
//            .recip()
//            .asinh()
//            .acosh()
//            .atanh()
//    };
//
//    // For reference and verification, in Julia the equivalent function is:
//    // f(x) = atanh(acosh(asinh(1/(tanh(cosh(sinh(atan(atan(acos(asin(tan(log10(log2(abs(log(x + 2.1, sqrt(abs(log(abs(cos(sin(x*2 - (x/0.5) * (x % 3) - x*1.1 + 1.2))))))^(x + 6))))))))), x + 1))))^2))))
//
//    let df = f(F3::var(x));
//
//    // The following test values were acquired using the ForwardDiff.jl Julia package.
//    println!("f(0.1) = {:?}", df.value());
//    assert_relative_eq!(df.value(), 0.4552286984773363, max_relative = 1e-10);
//    assert_relative_eq!(df.deriv().value(), 0.5090300997496865, max_relative = 1e-10);
//    assert_relative_eq!(
//        df.deriv().deriv().value(),
//        -0.24980963597585365,
//        max_relative = 1e-10
//    );
//    assert_relative_eq!(
//        df.deriv().deriv().deriv(),
//        0.2618169445307168,
//        max_relative = 1e-10
//    );
//}
