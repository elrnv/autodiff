// Multivariate quadratic function differentiation.

use autofloat::*;

fn main() {
    let x: f64 = 0.0;
    let y: f64 = 0.0;
    let f = |x: AutoFloat2<f64>, y: AutoFloat2<f64>| -> AutoFloat2<f64> {
        (x - 1.0f64) * (2. * y - 1.0f64)
    };

    let df = f(AutoFloat::variable(x, 0), AutoFloat::variable(y, 1));

    println!("f(x,y) = (x-1)*(2y-1)");
    println!(
        "df/dx = {} and df/dy = {} at x = {}, y = {}",
        df.dx[0], df.dx[1], x, y,
    );
}
