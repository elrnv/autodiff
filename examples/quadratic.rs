use autofloat::*;

fn main() {
    let x: f64 = 0.0;
    let f = |x: F1| -> F1 { (x - 1.0f64).pow(2.0) };

    let dfdx = f(F1::var(x));

    println!("f(x) = (x-1)^2");
    println!("df/dx = {} at x = {}", dfdx.deriv(), x,);
}
