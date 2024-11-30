use autofloat::{AutoFloat, AutoFloat2};
use num_traits::One;

// Define some target function for which we want to compute the derivative.
// This variant types the parameters explicitly, so it can only be used with AutoFloats of the exact same type.
fn quadratic_func(x: AutoFloat2<f64>, y: AutoFloat2<f64>) -> AutoFloat2<f64> {
    (x - AutoFloat::one()) * (AutoFloat::constant(2.0) * y - AutoFloat::one())
}

fn main() {
    // Use AutoFloat2 because we use a 2-dimensional function and we want a 2-dimensional gradient.
    // The first parameter determines the value of the variable.
    // The second prameter determines the index of the derivative for this variable within the gradient vector.
    let x = AutoFloat2::variable(2.25, 0);
    let y = AutoFloat2::variable(-1.75, 1);

    let result = quadratic_func(x, y);

    println!(
        "result={} gradient_x={} gradient_y={}",
        result.x, result.dx[0], result.dx[1]
    );
}
