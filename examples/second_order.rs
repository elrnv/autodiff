use autodiff::*;

fn main() {
    let x: f64 = 0.0;
    let f = |x: F2| -> F2 { (x - 1.0f64).pow(4.0) };

    let df = f(F2::var(x));

    println!("f(x) = (x-1)^4");
    println!("df/dx = {} at x = {}", df.deriv().value(), x);
    println!("d^2f/dx^2 = {} at x = {}", df.deriv().deriv(), x);
}
