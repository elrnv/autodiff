// Multivariate quadratic function differentiation.

use autofloat::*;

fn main() {
    let x: f64 = 0.0;
    let y: f64 = 0.0;
    let f = |x: F1, y: F1| -> F1 { (x - 1.0f64) * (2. * y - 1.0f64) };

    let dfdx = f(F1::var(x), F::cst(y));
    let dfdy = f(F::cst(x), F1::var(y));

    println!("f(x,y) = (x-1)*(2y-1)");
    println!(
        "df/dx = {} and df/dy = {} at x = {}, y = {}",
        dfdx.deriv(),
        dfdy.deriv(),
        x,
        y,
    );
}
