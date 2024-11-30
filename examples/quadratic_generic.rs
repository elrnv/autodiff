use autofloat::AutoFloat2;
use num_traits::float::FloatCore;

// Define some target function for which we want to compute the derivative.
// This variant is generic in T, but you could also use the `AutoFloat` type directly.
fn quadratic_func<T>(x: T, y: T) -> T
where
    T: FloatCore,
{
    (x - T::one()) * (T::from(2).unwrap() * y - T::one())
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
